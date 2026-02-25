from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field

import torch


@dataclass
class LoRAEmaTeacher:
    """EMA tracker + in-place swap context manager for adapter (LoRA) weights.

    Design constraints:
      - Graph-safe for torch.compile: never replace Parameter objects, only copy_ values.
      - DDP-safe: all ranks perform identical local copies; no syncing required.
      - Low overhead: EMA buffers are float32 and kept on the same device as the adapter params.
    """

    decay: float
    _ema_fp32: dict[str, torch.Tensor] = field(default_factory=dict, init=False)
    _backup: dict[str, torch.Tensor] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        if not (0.0 < float(self.decay) < 1.0):
            raise ValueError("LoRAEmaTeacher.decay must be in range (0, 1)")

    def init_from(self, network: torch.nn.Module) -> None:
        """Initialize EMA buffers from current network parameters."""
        self._ema_fp32.clear()
        self._backup.clear()

        for name, param in network.named_parameters():
            if not param.requires_grad:
                continue
            # Keep EMA in float32 for stability; keep it on the same device as params for fast swapping.
            self._ema_fp32[name] = param.detach().to(dtype=torch.float32).clone()
            self._backup[name] = torch.empty_like(param.detach())

    def update(self, network: torch.nn.Module) -> None:
        """Update EMA buffers from current network parameters (in-place)."""
        if len(self._ema_fp32) == 0:
            raise RuntimeError("LoRAEmaTeacher.update() called before init_from().")

        decay = float(self.decay)
        one_minus = 1.0 - decay

        with torch.no_grad():
            for name, param in network.named_parameters():
                if not param.requires_grad:
                    continue
                ema = self._ema_fp32.get(name, None)
                if ema is None:
                    raise KeyError(f"EMA buffer missing for parameter: {name}")
                # Keep EMA on the same device as the live param if devices change (rare, but safe).
                if ema.device != param.device:
                    ema = ema.to(device=param.device)
                    self._ema_fp32[name] = ema

                # ema = decay * ema + (1 - decay) * param
                ema.mul_(decay).add_(param.detach().to(dtype=torch.float32), alpha=one_minus)

    @contextmanager
    def apply_to(self, network: torch.nn.Module):
        """Temporarily swap network parameters to EMA weights (in-place), then restore."""
        if len(self._ema_fp32) == 0:
            raise RuntimeError("LoRAEmaTeacher.apply_to() called before init_from().")

        with torch.no_grad():
            for name, param in network.named_parameters():
                if not param.requires_grad:
                    continue
                backup = self._backup.get(name, None)
                ema = self._ema_fp32.get(name, None)
                if backup is None or ema is None:
                    raise KeyError(f"EMA/backup buffer missing for parameter: {name}")

                # Ensure backup matches current param tensor layout
                if backup.shape != param.shape or backup.dtype != param.dtype or backup.device != param.device:
                    backup = torch.empty_like(param.detach())
                    self._backup[name] = backup

                if ema.device != param.device:
                    ema = ema.to(device=param.device)
                    self._ema_fp32[name] = ema

                backup.copy_(param)
                param.copy_(ema.to(dtype=param.dtype))

        try:
            yield
        finally:
            with torch.no_grad():
                for name, param in network.named_parameters():
                    if not param.requires_grad:
                        continue
                    backup = self._backup.get(name, None)
                    if backup is None:
                        raise KeyError(f"Backup buffer missing for parameter: {name}")
                    param.copy_(backup)
