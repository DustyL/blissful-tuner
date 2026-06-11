# LoRA network module: currently conv2d is not fully supported
# reference:
# https://github.com/microsoft/LoRA/blob/main/loralib/layers.py
# https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/lora.py

import ast
import json
import math
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type, Union
import torch
import torch.nn as nn

from blissful_tuner.blissful_logger import BlissfulLogger
from musubi_tuner.networks.dora_utils import FP8_DTYPES, dequantize_fp8_weight
from musubi_tuner.networks.dora_utils import dora_weight_norm_materialized as _dora_weight_norm_materialized

if TYPE_CHECKING:
    from transformers import CLIPTextModel
else:
    CLIPTextModel = Any

logger = BlissfulLogger(__name__, "green")

RANK_PATTERN_MATCH_SEMANTICS = "original_name_fullmatch_first_match"


def parse_bool_arg(value, default: bool = False) -> bool:
    """Parse bool from network_args (handles string/bool/int)

    Supports explicit true tokens: "true", "1", "yes", "on"
    Supports explicit false tokens: "false", "0", "no", "off"
    Falls back to default only for None or unknown strings.
    """
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lower = value.lower()
        if lower in ("true", "1", "yes", "on"):
            return True
        if lower in ("false", "0", "no", "off"):
            return False
        # Unknown string - warn and use default
        logger.warning(f"Unknown bool value '{value}', using default={default}")
    return default


_PISSA_NITER_RE = re.compile(r"^pissa_niter_(\d+)$")


def parse_init_lora_weights_arg(value: Any = None) -> str:
    if value is None:
        return "kaiming"
    if value is True:
        return "kaiming"
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "true":
            return "kaiming"
        if normalized in ("kaiming", "orthogonal", "pissa"):
            return normalized
        m = _PISSA_NITER_RE.match(normalized)
        if m is not None:
            niter = int(m.group(1))
            if niter <= 0:
                raise ValueError(
                    f"init_lora_weights={value!r}: niter must be a positive integer (got {niter}). Example: 'pissa_niter_5'."
                )
            return normalized
    raise ValueError(
        f"init_lora_weights must be one of 'kaiming', 'orthogonal', 'pissa', "
        f"'pissa_niter_<N>' (positive int), or 'true' (alias for kaiming), got {value!r}"
    )


def parse_rank_or_alpha_pattern_arg(value: Any = None, kind: str = "rank_pattern") -> Optional[Dict[re.Pattern, Union[int, float]]]:
    if kind not in ("rank_pattern", "alpha_pattern"):
        raise ValueError(f"kind must be 'rank_pattern' or 'alpha_pattern', got {kind!r}")
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"{kind} must be a JSON or Python dict literal, got {value!r}: {e}") from e
    if not isinstance(value, dict):
        raise ValueError(f"{kind} must parse to a dict, got {type(value).__name__}")

    parsed = {}
    for raw_pattern, raw_val in value.items():
        if not isinstance(raw_pattern, str):
            raise ValueError(f"{kind} keys must be regex strings, got {raw_pattern!r}")
        try:
            compiled = re.compile(raw_pattern)
        except re.error as e:
            raise ValueError(f"{kind} regex {raw_pattern!r} did not compile: {e}") from e

        if kind == "rank_pattern":
            if isinstance(raw_val, bool) or not isinstance(raw_val, int) or raw_val <= 0:
                raise ValueError(f"{kind} value for {raw_pattern!r} must be a positive int, got {raw_val!r}")
        else:
            if isinstance(raw_val, bool) or not isinstance(raw_val, (int, float)) or raw_val <= 0:
                raise ValueError(f"{kind} value for {raw_pattern!r} must be a positive number, got {raw_val!r}")
        parsed[compiled] = raw_val
    return parsed


def _match_pattern_value(
    patterns: Optional[Dict[re.Pattern, Union[int, float]]],
    original_name: str,
) -> tuple[Optional[Union[int, float]], Optional[str]]:
    if not patterns:
        return None, None
    for pattern, value in patterns.items():
        if pattern.fullmatch(original_name):
            return value, pattern.pattern
    return None, None


def _pattern_metadata_json(patterns: Dict[re.Pattern, Union[int, float]]) -> str:
    return json.dumps({pattern.pattern: value for pattern, value in patterns.items()})


def _init_kaiming_lora_pair(lora_down: torch.nn.Module, lora_up: torch.nn.Module) -> None:
    torch.nn.init.kaiming_uniform_(lora_down.weight, a=math.sqrt(5))
    torch.nn.init.zeros_(lora_up.weight)


def _init_orthogonal_lora_pair(
    lora_down: torch.nn.Module,
    lora_up: torch.nn.Module,
    in_features: int,
    out_features: int,
    rank: int,
) -> None:
    """Orthogonal LoRA init from PEFT.

    This is Linear-only QR-based init. It preserves the standard LoRA
    zero-delta-at-init invariant because B @ A is zero, while both A and B
    start nonzero.
    """
    if rank % 2 != 0:
        raise ValueError(
            f"Orthogonal LoRA init requires even rank, got {rank}. "
            "Either set --network_dim to an even number or use init_lora_weights=kaiming."
        )

    dtype = lora_down.weight.dtype
    device = lora_down.weight.device
    with torch.no_grad():
        x = torch.randn(rank, rank, device=device, dtype=torch.float32)
        q, _ = torch.linalg.qr(x)
        q_odd = q[0::2, :]
        q_even = q[1::2, :]

        new_down = (torch.randn(in_features, rank // 2, device=device, dtype=torch.float32).mm(q_odd).T / 10.0).contiguous()
        new_up = (torch.randn(rank // 2, out_features, device=device, dtype=torch.float32).T.mm(q_even) / 10.0).contiguous()

        lora_down.weight.copy_(new_down.to(dtype=dtype))
        lora_up.weight.copy_(new_up.to(dtype=dtype))


_PISSA_ALLOWED_BASE_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _init_pissa_lora_pair(
    lora_down: torch.nn.Module,
    lora_up: torch.nn.Module,
    base_weight: torch.Tensor,
    rank: int,
    scaling: float,
    mode: str,
) -> torch.Tensor:
    """Initialize lora_down (A) and lora_up (B) via PiSSA decomposition.

    Mirrors PEFT's pissa_init at peft/src/peft/tuners/lora/layer.py:360-393.
    The math uses SVD of the base weight to seed lora_A/lora_B with the
    principal-rank-r components, then returns the residual weight tensor
    so the caller can mutate org_module.weight.data with it.

    Returning the residual (rather than mutating in place) keeps this
    helper a pure function — easier to test in isolation against PEFT
    for math parity. The caller (LoRAModule.__init__, in commit 2) is
    responsible for the in-place base-weight mutation.

    Forward equivalence at step 0 holds because:
        original_weight == residual + scaling * lora_B @ lora_A

    Args:
        lora_down: LoRA A module (the down-projection). Modified in place.
        lora_up: LoRA B module (the up-projection). Modified in place.
        base_weight: The base layer's weight tensor. NOT mutated; the
            residual is returned for the caller to write back.
        rank: LoRA rank (must equal lora_down.out_features and
            lora_up.in_features after this init).
        scaling: The LoRA scale (alpha/r for standard, alpha/sqrt(r) for
            rsLoRA). PEFT's pissa_init divides Sr by scaling so the
            forward-equivalence reconstruction holds.
        mode: "pissa" (full SVD) or "pissa_niter_<N>" (Newton-Schulz
            approximate SVD via torch.svd_lowrank, faster for big weights).

    Returns:
        The residual weight tensor (same shape, dtype, device as base_weight).
        Caller mutates `org_module.weight.data = returned_residual`.

    Raises:
        TypeError: when base_weight dtype is not fp32/fp16/bf16 (PiSSA
            cannot run on fp8/int8 bases — SVD is unstable).
        ValueError: when mode does not match "pissa" or "pissa_niter_<N>".
        NotImplementedError: when base_weight is not 2D (Conv2d 4D weights
            are not supported; PEFT's reference also crashes on them).
    """
    if base_weight.dtype not in _PISSA_ALLOWED_BASE_DTYPES:
        raise TypeError(
            f"PiSSA initialization requires base weight in float32/float16/bfloat16, "
            f"got {base_weight.dtype}. Re-quantize after PiSSA init if you need fp8."
        )
    if base_weight.dim() != 2:
        raise NotImplementedError(
            f"PiSSA initialization requires a 2D base weight (Linear). "
            f"Got {base_weight.dim()}D weight (likely Conv2d). PEFT's pissa_init also "
            f"crashes on 4D weights — this is a math limitation, not a missing "
            f"implementation. Use init_lora_weights=kaiming for non-Linear modules."
        )

    target_dtype = lora_down.weight.dtype
    target_device = lora_down.weight.device

    # PEFT does the SVD in fp32 regardless of base dtype, then casts back.
    weight_fp32 = base_weight.to(torch.float32)

    if mode == "pissa":
        V, S, Uh = torch.linalg.svd(weight_fp32, full_matrices=False)
        Vr = V[:, :rank]
        Sr = S[:rank].clone()  # clone so the in-place /= doesn't touch the SVD output
        Sr /= scaling
        Uhr = Uh[:rank, :]
    else:
        m = _PISSA_NITER_RE.match(mode)
        if m is None:
            raise ValueError(f"PiSSA mode must be 'pissa' or 'pissa_niter_<N>', got {mode!r}.")
        niter = int(m.group(1))
        Vr, Sr, Ur = torch.svd_lowrank(weight_fp32, rank, niter=niter)
        Sr = Sr / scaling
        Uhr = Ur.t()

    sqrt_sr = torch.sqrt(Sr)
    lora_A = torch.diag(sqrt_sr) @ Uhr  # (rank, in_features)
    lora_B = Vr @ torch.diag(sqrt_sr)  # (out_features, rank)

    # Residual: original - scaling * (B @ A) so forward(residual) + scale*forward(LoRA) == forward(original)
    residual_fp32 = weight_fp32 - scaling * lora_B @ lora_A

    with torch.no_grad():
        lora_down.weight.copy_(lora_A.to(device=target_device, dtype=target_dtype))
        lora_up.weight.copy_(lora_B.to(device=target_device, dtype=target_dtype))

    return residual_fp32.to(device=base_weight.device, dtype=base_weight.dtype)


HUNYUAN_TARGET_REPLACE_MODULES = ["MMDoubleStreamBlock", "MMSingleStreamBlock"]


class DoRALayer(nn.Module):
    """Handles DoRA magnitude computation for Linear layers only"""

    def __init__(self, out_features: int):
        super().__init__()
        # Magnitude vector - initialized to 1s, will be set properly in update_layer
        self.weight = nn.Parameter(torch.ones(out_features))

    def get_weight_norm_materialized(
        self,
        weight: torch.Tensor,
        lora_weight: torch.Tensor,
        scaling: float,
        eps: float = 1e-6,
        scale_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate row-wise L2 norm by materializing B@A (for init/merge only).

        For [out, in] weight matrix, computes ||row_i||_2 for each output row.
        Pass the module's ``scale_weight`` buffer when the base weight is fp8-prequantized.
        """
        return _dora_weight_norm_materialized(weight, lora_weight, scaling, eps, scale_weight=scale_weight)

    def get_weight_norm_efficient(
        self,
        W: torch.Tensor,
        A: torch.Tensor,
        B: torch.Tensor,
        s: float,
        eps: float = 1e-6,
        scale_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Calculate row-wise L2 norm WITHOUT materializing B@A (memory efficient).

        For [out, in] weight matrix, computes ||row_i||_2 for each output row.
        For each output row i: ||W_i + s*(B_i @ A)||^2 = ||W_i||^2 + 2s*<W_i, B_i@A> + s^2*||B_i@A||^2

        Args:
            W: [out, in] base weight
            A: [r, in] lora_down.weight
            B: [out, r] lora_up.weight
            s: scaling factor (multiplier * scale)
            scale_weight: the module's fp8 scale buffer; required when W is fp8-prequantized
        """
        # fp8 base: dequantize to true weights first. A bare .float() on raw fp8 would NOT crash but
        # yields quantization-lattice magnitudes (~200x off) — silently wrong norms, wrong DoRA scaling.
        # After dequant W is float32, so the .to(W.dtype) at the end also stays non-fp8 (an fp8 norm
        # would crash the downstream magnitude/norm division with the same Float8-promotion error).
        if W.dtype in FP8_DTYPES:
            W = dequantize_fp8_weight(W, scale_weight)
        Wf, Af, Bf = W.float(), A.float(), B.float()

        # ||W_i||^2 for each row
        w_norm2 = (Wf * Wf).sum(dim=1)  # [out]

        # Cross term: 2s * <W_i, B_i @ A> = 2s * sum_j(W_ij * (B_i @ A)_j)
        # Efficient: (A @ W.T) gives [r, out], then (B * (A @ W.T).T).sum(dim=1)
        AWt = Af @ Wf.T  # [r, out]
        cross = 2.0 * s * (Bf * AWt.T).sum(dim=1)  # [out]

        # LoRA norm term: s^2 * ||B_i @ A||^2
        # ||B_i @ A||^2 = B_i @ (A @ A.T) @ B_i.T
        G = Af @ Af.T  # [r, r]
        BG = Bf @ G  # [out, r]
        lora_norm2 = (BG * Bf).sum(dim=1)  # [out]

        # Combine: ||W + s*BA||^2 = w_norm2 + cross + s^2*lora_norm2
        norm_squared = w_norm2 + cross + (s * s) * lora_norm2
        return norm_squared.clamp_min(eps).sqrt().to(W.dtype)

    def update_layer(
        self,
        base_weight: torch.Tensor,
        lora_A: torch.Tensor,
        lora_B: torch.Tensor,
        scaling: float,
        scale_weight: Optional[torch.Tensor] = None,
    ) -> None:
        """Initialize magnitude from base weights + LoRA (materializes B@A, called once)"""
        with torch.no_grad():
            lora_weight = lora_B @ lora_A
            weight_norm = self.get_weight_norm_materialized(base_weight, lora_weight, scaling, scale_weight=scale_weight)
            # Use copy_() to preserve parameter identity for optimizer/FSDP
            self.weight.copy_(weight_norm.detach())


class LoRAModule(torch.nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        dropout=None,
        rank_dropout=None,
        module_dropout=None,
        split_dims: Optional[List[int]] = None,
        use_rslora: bool = False,
        use_dora: bool = False,
        init_lora_weights: str = "kaiming",
    ):
        """
        if alpha == 0 or None, alpha is rank (no scaling).

        split_dims is used to mimic the split qkv of multi-head attention.
        """
        super().__init__()
        self.lora_name = lora_name
        init_lora_weights = parse_init_lora_weights_arg(init_lora_weights)
        self.init_lora_weights = init_lora_weights
        self.orthogonal_init_fallback_reason = None

        if org_module.__class__.__name__ == "Conv2d":
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels
        else:
            in_dim = org_module.in_features
            out_dim = org_module.out_features

        self.lora_dim = lora_dim
        self.split_dims = split_dims

        # alpha/scale moved earlier (was below init dispatch) so PiSSA can read self.scale
        # for its SVD divide step. Kaiming/orthogonal don't depend on self.scale, so this
        # move is semantically a no-op for those paths — pinned by TestAlphaScaleBackCompatAfterMove.
        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().float().numpy()  # without casting, bf16 causes error

        self.use_rslora = use_rslora

        # Handle alpha==0/None: means "no scaling" (scale=1)
        # NOTE: For RS-LoRA with alpha=0, we store alpha=sqrt(r) so scale=1. This computed
        # alpha is what gets saved in weights. External tools that ignore use_rslora_flag
        # and assume alpha/r scaling will misinterpret it. The use_rslora_flag buffer in
        # saved weights indicates the correct interpretation.
        if alpha is None or alpha == 0:
            alpha = math.sqrt(self.lora_dim) if use_rslora else self.lora_dim

        self.scale = alpha / math.sqrt(self.lora_dim) if use_rslora else alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # for save/load

        # PiSSA fail-fast rejects: fire before any layer allocation so we don't silently
        # disable a user's intended init choice. Gates on the user's intent (init_lora_weights
        # / use_dora arg), not on the post-disable self.use_dora.
        is_pissa = init_lora_weights.startswith("pissa")
        if is_pissa:
            if org_module.__class__.__name__ == "Conv2d":
                raise ValueError(
                    f"init_lora_weights={init_lora_weights!r} is not supported for Conv2d module {lora_name!r}. "
                    "PEFT's pissa_init also crashes on 4D weights — this is a math limitation, "
                    "not a missing implementation. Use init_lora_weights=kaiming for Conv2d modules, "
                    "or restrict --include_patterns to skip them."
                )
            if split_dims is not None:
                raise ValueError(
                    f"init_lora_weights={init_lora_weights!r} is not supported with split_dims for module {lora_name!r}. "
                    "Per-split SVD doesn't have a single canonical interpretation in v1. "
                    "Use init_lora_weights=kaiming or train without split_dims for fused-QKV PiSSA."
                )
            if use_dora:
                raise ValueError(
                    f"init_lora_weights={init_lora_weights!r} cannot be combined with use_dora=True "
                    f"for module {lora_name!r}. PEFT mixes the two only via a separate "
                    "pissa_decompose_dora utility (deferred to Tier 2 #6d). "
                    "Pick one of init_lora_weights=pissa OR use_dora=True for v1."
                )

        if split_dims is None:
            if org_module.__class__.__name__ == "Conv2d":
                kernel_size = org_module.kernel_size
                stride = org_module.stride
                padding = org_module.padding
                self.lora_down = torch.nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
                self.lora_up = torch.nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)
            else:
                self.lora_down = torch.nn.Linear(in_dim, self.lora_dim, bias=False)
                self.lora_up = torch.nn.Linear(self.lora_dim, out_dim, bias=False)

            if is_pissa:
                # PiSSA: SVD the base, residualize. Conv2d / split_dims / DoRA already rejected above.
                residual = _init_pissa_lora_pair(
                    self.lora_down,
                    self.lora_up,
                    org_module.weight.data,
                    self.lora_dim,
                    self.scale,
                    init_lora_weights,
                )
                with torch.no_grad():
                    org_module.weight.data.copy_(residual)
            elif init_lora_weights == "orthogonal" and org_module.__class__.__name__ != "Conv2d":
                _init_orthogonal_lora_pair(self.lora_down, self.lora_up, in_dim, out_dim, self.lora_dim)
            else:
                if init_lora_weights == "orthogonal":
                    self.orthogonal_init_fallback_reason = "Conv2d"
                _init_kaiming_lora_pair(self.lora_down, self.lora_up)
        else:
            # conv2d not supported
            assert sum(split_dims) == out_dim, "sum of split_dims must be equal to out_dim"
            assert org_module.__class__.__name__ == "Linear", "split_dims is only supported for Linear"
            # print(f"split_dims: {split_dims}")
            self.lora_down = torch.nn.ModuleList(
                [torch.nn.Linear(in_dim, self.lora_dim, bias=False) for _ in range(len(split_dims))]
            )
            self.lora_up = torch.nn.ModuleList([torch.nn.Linear(self.lora_dim, split_dim, bias=False) for split_dim in split_dims])
            for lora_down, lora_up, split_dim in zip(self.lora_down, self.lora_up, split_dims):
                if init_lora_weights == "orthogonal":
                    _init_orthogonal_lora_pair(lora_down, lora_up, in_dim, split_dim, self.lora_dim)
                else:
                    _init_kaiming_lora_pair(lora_down, lora_up)

        # same as microsoft's
        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        # Adapter enable flag: LoRANetwork.set_enabled(...) propagates here, and forward() consults
        # it. Mirrors the LoRAInfModule attribute (lora.py:666). Default True so the standard
        # training forward unchanged when set_enabled is never called.
        self.enabled = True

        # Detect layer type - DoRA only for Linear (not Conv)
        is_linear = org_module.__class__.__name__ == "Linear"

        # DoRA disabled if: not Linear, split_dims used, or dropout/rank_dropout actively used
        # Note: dropout=0.0 is treated as disabled (common in configs)
        dora_dropout_conflict = (dropout is not None and dropout > 0) or (rank_dropout is not None and rank_dropout > 0)
        self.use_dora = use_dora and is_linear and split_dims is None and not dora_dropout_conflict
        self.dora_layer = None
        self.dora_disabled_reason = None  # Track reason for summary logging

        if self.use_dora:
            self.dora_layer = DoRALayer(out_dim)
        elif use_dora:
            # Store reason for DoRA being disabled (for summary logging in network)
            if not is_linear:
                self.dora_disabled_reason = "non-Linear"
            elif split_dims is not None:
                self.dora_disabled_reason = "split_dims"
            elif dora_dropout_conflict:
                self.dora_disabled_reason = "dropout"

    def apply_to(self):
        self.org_forward = self.org_module.forward
        # Initialize DoRA magnitude from current base weights
        if self.use_dora and self.dora_layer is not None:
            self._init_dora()
        self.org_module.forward = self.forward
        del self.org_module

    def _init_dora(self):
        """Initialize DoRA magnitude vector from current weights"""
        base_layer = self.org_module
        base_weight = base_layer.weight.data
        lora_A = self.lora_down.weight
        lora_B = self.lora_up.weight
        # fp8-prequantized base (e.g. Ideogram 4): the monkey-patch's scale_weight buffer is needed
        # to dequantize raw fp8 values to true weights; absent on ordinary bf16/fp32 modules.
        scale_weight = getattr(base_layer, "scale_weight", None)
        self.dora_layer.update_layer(base_weight, lora_A, lora_B, self.scale * self.multiplier, scale_weight=scale_weight)

    def _get_base_layer(self):
        """Get base layer (for live weight and bias access)"""
        return self.org_forward.__self__

    def _dora_delta(self, base_result: torch.Tensor, lora_result: torch.Tensor, scale: float) -> torch.Tensor:
        """
        Compute DoRA delta to add to base output.

        Uses memory-efficient weight_norm computation (no B@A materialization).
        Excludes bias from (mag_norm_scale - 1) term per PEFT/paper.
        """
        base_layer = self._get_base_layer()
        W = base_layer.weight
        A = self.lora_down.weight
        B = self.lora_up.weight
        s = scale * self.multiplier

        # Project convention: multiplier=0 should be a true no-op.
        # DoRA can otherwise rescale the base output even when s==0.
        if s == 0:
            return torch.zeros_like(base_result)

        # Weight norm using efficient computation (no_grad to avoid building compute graph)
        # Detached per DoRA paper section 4.3 - gradients don't flow through the norm
        with torch.no_grad():
            weight_norm = self.dora_layer.get_weight_norm_efficient(
                W, A, B, s, scale_weight=getattr(base_layer, "scale_weight", None)
            )

        # Magnitude / norm scaling - reshape based on output tensor dimensions
        # 2D: [batch, out] -> mag_norm_scale shape [1, out]
        # 3D: [batch, seq, out] -> mag_norm_scale shape [1, 1, out]
        mag_norm_scale = self.dora_layer.weight / weight_norm
        if base_result.ndim == 2:
            mag_norm_scale = mag_norm_scale.view(1, -1)
        elif base_result.ndim == 3:
            mag_norm_scale = mag_norm_scale.view(1, 1, -1)
        else:
            # Fallback: reshape to broadcast on last dim
            shape = [1] * (base_result.ndim - 1) + [-1]
            mag_norm_scale = mag_norm_scale.view(*shape)

        # PEFT-style bias handling: exclude bias from the (mag_norm_scale - 1) term
        # base_result includes bias, so subtract it for that term, then it comes back via org_forwarded
        bias = base_layer.bias
        if bias is not None:
            base_wo_bias = base_result - bias
        else:
            base_wo_bias = base_result

        # DoRA delta formula
        # delta = (m/norm - 1) * base_wo_bias + (m/norm) * lora_out * s
        delta = (mag_norm_scale - 1) * base_wo_bias + mag_norm_scale * lora_result * s
        # Preserve the module output dtype - DoRA magnitude may be float32,
        # which would otherwise promote Linear outputs and break attention backends
        # (e.g., CuTE FlashAttention requires q.dtype == k.dtype == v.dtype).
        return delta.to(dtype=base_result.dtype)

    def forward(self, x):
        org_forwarded = self.org_forward(x)

        # Project convention: multiplier=0 should be a true no-op.
        if self.multiplier == 0:
            return org_forwarded

        # Project convention: set_enabled(False) on the parent LoRANetwork (lora.py:1309) must produce
        # a true no-op forward. LoRAInfModule.forward checks this at line 844 for inference; the
        # training forward MUST do the same so prior_model_context (which calls set_enabled(False)
        # around the teacher pass in mask_loss_process_batch.py + ideogram4_train_network.py) actually
        # disables the LoRA contribution. Without this check, the teacher forward computes against
        # the LoRA-active model, prior_loss_unreduced = MSE(student, teacher) = 0, and prior
        # preservation silently does nothing — confirmed empirically in DLAY v5 training where
        # masked_loss/prior = 0 across all 2145 steps despite prior_weight=0.5.
        if not self.enabled:
            return org_forwarded

        # module dropout
        if self.module_dropout is not None and self.training:
            if torch.rand(1) < self.module_dropout:
                return org_forwarded

        # Explicit dtype harmonization for the LoRA matmul. Under autocast (the common case)
        # these casts are no-ops because autocast already inserted them at the F.linear boundary.
        # Without autocast (e.g., Ideogram 4 disables it at do_inference / process_batch to keep
        # training-inference parity — see src/musubi_tuner/utils/accelerate_utils.py), the base
        # model's input dtype (bf16) and the LoRA weight dtype (fp32 by default) don't match, and
        # F.linear raises "expected mat1 and mat2 to have the same dtype". The explicit cast makes
        # LoRA forward correct regardless of the caller's autocast state.
        if self.split_dims is None:
            lora_dtype = self.lora_down.weight.dtype
            lx = self.lora_down(x.to(lora_dtype) if x.dtype != lora_dtype else x)

            # normal dropout
            if self.dropout is not None and self.training:
                lx = torch.nn.functional.dropout(lx, p=self.dropout)

            # rank dropout
            if self.rank_dropout is not None and self.training:
                mask = torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout
                if len(lx.size()) == 3:
                    mask = mask.unsqueeze(1)  # for Text Encoder
                elif len(lx.size()) == 4:
                    mask = mask.unsqueeze(-1).unsqueeze(-1)  # for Conv2d
                lx = lx * mask

                # scaling for rank dropout: treat as if the rank is changed
                scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
            else:
                scale = self.scale

            lx = self.lora_up(lx)

            if self.use_dora and self.dora_layer is not None:
                # DoRA: compute delta and add to base. Cast lora_result to base's dtype so the
                # mixed-precision math inside _dora_delta is consistent and the output preserves
                # the module's expected dtype.
                if scale * self.multiplier == 0:
                    return org_forwarded
                dora_delta = self._dora_delta(org_forwarded, lx.to(org_forwarded.dtype), scale)
                return org_forwarded + dora_delta
            else:
                # Cast back to base's dtype before adding so addition doesn't promote the
                # output dtype (e.g., fp32 LoRA output + bf16 base would otherwise become fp32).
                return org_forwarded + lx.to(org_forwarded.dtype) * self.multiplier * scale
        else:
            # All sub-loras share the same dtype (created together with the same nn.Linear default).
            lora_dtype = self.lora_down[0].weight.dtype
            x_for_lora = x.to(lora_dtype) if x.dtype != lora_dtype else x
            lxs = [lora_down(x_for_lora) for lora_down in self.lora_down]

            # normal dropout
            if self.dropout is not None and self.training:
                lxs = [torch.nn.functional.dropout(lx, p=self.dropout) for lx in lxs]

            # rank dropout
            if self.rank_dropout is not None and self.training:
                masks = [torch.rand((lx.size(0), self.lora_dim), device=lx.device) > self.rank_dropout for lx in lxs]
                for i in range(len(lxs)):
                    # `lx` from the comprehension above doesn't leak into this scope in Py3,
                    # so we must consult lxs[i] directly. Use .dim() (returns int) — the prior
                    # `len(lx.size()) == 3` form raised UnboundLocalError, and `len(lxs[i].dim())`
                    # would raise TypeError because dim() is an int.
                    ndim = lxs[i].dim()
                    if ndim == 3:
                        masks[i] = masks[i].unsqueeze(1)
                    elif ndim == 4:
                        masks[i] = masks[i].unsqueeze(-1).unsqueeze(-1)
                    lxs[i] = lxs[i] * masks[i]

                # scaling for rank dropout: treat as if the rank is changed
                scale = self.scale * (1.0 / (1.0 - self.rank_dropout))  # redundant for readability
            else:
                scale = self.scale

            lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]

            # Cast concatenated LoRA result back to base's dtype to preserve output dtype.
            return org_forwarded + torch.cat(lxs, dim=-1).to(org_forwarded.dtype) * self.multiplier * scale


class LoRAInfModule(LoRAModule):
    def __init__(
        self,
        lora_name,
        org_module: torch.nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ):
        # no dropout for inference
        super().__init__(lora_name, org_module, multiplier, lora_dim, alpha, use_rslora=use_rslora, use_dora=use_dora)

        self.org_module_ref = [org_module]  # for reference
        self.enabled = True
        self.network: LoRANetwork = None
        self._warned_uninit_dora_mag = False  # for warn-once in get_weight()

    def set_network(self, network):
        self.network = network

    # merge weight to org_module
    # def merge_to(self, sd, dtype, device, non_blocking=False):
    #     if torch.cuda.is_available():
    #         stream = torch.cuda.Stream(device=device)
    #         with torch.cuda.stream(stream):
    #             print(f"merge_to {self.lora_name}")
    #             self._merge_to(sd, dtype, device, non_blocking)
    #             torch.cuda.synchronize(device=device)
    #             print(f"merge_to {self.lora_name} done")
    #         torch.cuda.empty_cache()
    #     else:
    #         self._merge_to(sd, dtype, device, non_blocking)

    def merge_to(self, sd, dtype, device, non_blocking=False, safe_merge: bool = False):
        # Project convention: multiplier=0 should be a true no-op.
        if self.multiplier == 0:
            return

        # extract weight from org_module
        org_sd = self.org_module.state_dict()
        weight = org_sd["weight"]
        org_dtype = weight.dtype
        org_device = weight.device
        weight = weight.to(device, dtype=torch.float, non_blocking=non_blocking)  # for calculation

        if dtype is None:
            dtype = org_dtype
        if device is None:
            device = org_device

        if self.split_dims is None:
            # get up/down weight
            down_weight = sd["lora_down.weight"].to(device, dtype=torch.float, non_blocking=non_blocking)
            up_weight = sd["lora_up.weight"].to(device, dtype=torch.float, non_blocking=non_blocking)

            # merge weight
            if len(weight.size()) == 2:
                # linear
                if self.use_dora and "dora_layer.weight" in sd:
                    # DoRA merge - read magnitude from weights file, not from self.dora_layer
                    dora_magnitude = sd["dora_layer.weight"].to(device, dtype=torch.float, non_blocking=non_blocking)
                    delta_weight = self.multiplier * (up_weight @ down_weight) * self.scale
                    weight_norm = self.dora_layer.get_weight_norm_materialized(weight, delta_weight, 1.0)
                    dora_factor = dora_magnitude / weight_norm
                    weight = dora_factor.view(-1, 1) * (weight + delta_weight)
                else:
                    if self.use_dora and "dora_layer.weight" not in sd:
                        raise ValueError(
                            f"DoRA enabled for {self.lora_name} but dora_layer.weight is missing from weights. "
                            f"This would silently produce incorrect results (uninitialized magnitudes)."
                        )
                    weight = weight + self.multiplier * (up_weight @ down_weight) * self.scale
            elif down_weight.size()[2:4] == (1, 1):
                # conv2d 1x1
                weight = (
                    weight
                    + self.multiplier
                    * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                    * self.scale
                )
            else:
                # conv2d 3x3
                conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
                # logger.info(conved.size(), weight.size(), module.stride, module.padding)
                weight = weight + self.multiplier * conved * self.scale

            if safe_merge and not torch.isfinite(weight).all():
                raise ValueError(f"safe_merge detected non-finite merged weight for {self.lora_name}")

            # set weight to org_module
            org_sd["weight"] = weight.to(org_device, dtype=dtype)  # back to CPU without non_blocking
            self.org_module.load_state_dict(org_sd)
        else:
            # split_dims
            total_dims = sum(self.split_dims)
            for i in range(len(self.split_dims)):
                # get up/down weight
                down_weight = sd[f"lora_down.{i}.weight"].to(device, torch.float, non_blocking=non_blocking)  # (rank, in_dim)
                up_weight = sd[f"lora_up.{i}.weight"].to(device, torch.float, non_blocking=non_blocking)  # (split dim, rank)

                # pad up_weight -> (total_dims, rank)
                padded_up_weight = torch.zeros((total_dims, up_weight.size(0)), device=device, dtype=torch.float)
                padded_up_weight[sum(self.split_dims[:i]) : sum(self.split_dims[: i + 1])] = up_weight

                # merge weight
                weight = weight + self.multiplier * (up_weight @ down_weight) * self.scale

            if safe_merge and not torch.isfinite(weight).all():
                raise ValueError(f"safe_merge detected non-finite merged weight for {self.lora_name}")

            # set weight to org_module
            org_sd["weight"] = weight.to(org_device, dtype)  # back to CPU without non_blocking
            self.org_module.load_state_dict(org_sd)

    # return weight for merge
    def get_weight(self, multiplier=None):
        if multiplier is None:
            multiplier = self.multiplier

        # Project convention: multiplier=0 should be a true no-op.
        if multiplier == 0:
            return torch.zeros_like(self.org_module_ref[0].weight, dtype=torch.float)

        # get up/down weight from module
        up_weight = self.lora_up.weight.to(torch.float)
        down_weight = self.lora_down.weight.to(torch.float)

        # pre-calculated weight
        if len(down_weight.size()) == 2:
            # linear
            if self.use_dora and self.dora_layer is not None:
                # Guard: warn (once) if DoRA magnitude looks uninitialized (all-ones from __init__)
                # This suggests weights weren't loaded before calling get_weight()/pre_calculation()
                mag = self.dora_layer.weight
                if not self._warned_uninit_dora_mag:
                    with torch.no_grad():
                        ones = torch.ones_like(mag)
                        if torch.allclose(mag.detach(), ones, rtol=1e-5, atol=1e-5):
                            logger.warning(
                                f"DoRA magnitude for {self.lora_name} appears uninitialized (all ones). "
                                f"Ensure weights are loaded via load_state_dict() before calling pre_calculation(). "
                                f"DoRA deltas will be incorrect otherwise."
                            )
                            self._warned_uninit_dora_mag = True
                # Include DoRA scaling in pre-calculated weight (materialization OK here)
                # fp8 base: dequantize via scale_weight before the float cast — raw fp8 values are
                # not true weights and would silently mis-scale the merge.
                base_weight = dequantize_fp8_weight(
                    self.org_module_ref[0].weight.data, getattr(self.org_module_ref[0], "scale_weight", None)
                ).to(torch.float)
                lora_weight = multiplier * (up_weight @ down_weight) * self.scale
                weight_norm = self.dora_layer.get_weight_norm_materialized(base_weight, lora_weight, 1.0)
                dora_factor = mag.to(torch.float) / weight_norm
                merged = dora_factor.view(-1, 1) * (base_weight + lora_weight)
                return merged - base_weight  # Return delta only
            else:
                weight = multiplier * (up_weight @ down_weight) * self.scale
        elif down_weight.size()[2:4] == (1, 1):
            # conv2d 1x1
            weight = (
                multiplier
                * (up_weight.squeeze(3).squeeze(2) @ down_weight.squeeze(3).squeeze(2)).unsqueeze(2).unsqueeze(3)
                * self.scale
            )
        else:
            # conv2d 3x3
            conved = torch.nn.functional.conv2d(down_weight.permute(1, 0, 2, 3), up_weight).permute(1, 0, 2, 3)
            weight = multiplier * conved * self.scale

        return weight

    def default_forward(self, x):
        # logger.info(f"default_forward {self.lora_name} {x.size()}")
        # Project convention: multiplier=0 should be a true no-op.
        if self.multiplier == 0:
            return self.org_forward(x)
        if self.split_dims is None:
            lx = self.lora_down(x)
            lx = self.lora_up(lx)

            # DoRA path for inference
            if self.use_dora and self.dora_layer is not None:
                org_forwarded = self.org_forward(x)
                if self.multiplier * self.scale == 0:
                    return org_forwarded
                dora_delta = self._dora_delta(org_forwarded, lx, self.scale)
                return org_forwarded + dora_delta
            else:
                return self.org_forward(x) + lx * self.multiplier * self.scale
        else:
            lxs = [lora_down(x) for lora_down in self.lora_down]
            lxs = [lora_up(lx) for lora_up, lx in zip(self.lora_up, lxs)]
            return self.org_forward(x) + torch.cat(lxs, dim=-1) * self.multiplier * self.scale

    def forward(self, x):
        if not self.enabled:
            return self.org_forward(x)
        return self.default_forward(x)


def create_arch_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    # add default exclude patterns
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is None:
        exclude_patterns = []
    else:
        exclude_patterns = ast.literal_eval(exclude_patterns)

    # exclude if 'img_mod', 'txt_mod' or 'modulation' in the name
    exclude_patterns.append(r".*(img_mod|txt_mod|modulation).*")

    kwargs["exclude_patterns"] = exclude_patterns

    return create_network(
        HUNYUAN_TARGET_REPLACE_MODULES,
        "lora_unet",
        multiplier,
        network_dim,
        network_alpha,
        vae,
        text_encoders,
        unet,
        neuron_dropout=neuron_dropout,
        **kwargs,
    )


def create_network(
    target_replace_modules: List[str],
    prefix: str,
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: nn.Module,
    neuron_dropout: Optional[float] = None,
    module_class: Type[object] = None,
    module_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """architecture independent network creation"""
    if network_dim is None:
        network_dim = 4  # default
    if network_alpha is None:
        network_alpha = 1.0

    # extract dim/alpha for conv2d, and block dim
    conv_dim = kwargs.get("conv_dim", None)
    conv_alpha = kwargs.get("conv_alpha", None)
    if conv_dim is not None:
        conv_dim = int(conv_dim)
        if conv_alpha is None:
            conv_alpha = 1.0
        else:
            conv_alpha = float(conv_alpha)

    rank_pattern = parse_rank_or_alpha_pattern_arg(kwargs.get("rank_pattern", None), "rank_pattern")
    alpha_pattern = parse_rank_or_alpha_pattern_arg(kwargs.get("alpha_pattern", None), "alpha_pattern")

    # rank/module dropout
    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    # Defensive coercion for neuron_dropout (in case passed as string from wrapper)
    if neuron_dropout is not None:
        # Guard against "None"/"" strings that would fail float()
        if isinstance(neuron_dropout, str) and neuron_dropout.lower() in ("none", ""):
            neuron_dropout = None
        else:
            neuron_dropout = float(neuron_dropout)

    # verbose
    verbose = parse_bool_arg(kwargs.get("verbose", None), default=False)

    # regular expression for module selection: exclude and include
    exclude_patterns = kwargs.get("exclude_patterns", None)
    if exclude_patterns is not None and isinstance(exclude_patterns, str):
        exclude_patterns = ast.literal_eval(exclude_patterns)
    include_patterns = kwargs.get("include_patterns", None)
    if include_patterns is not None and isinstance(include_patterns, str):
        include_patterns = ast.literal_eval(include_patterns)

    # RS-LoRA and DoRA
    use_rslora = parse_bool_arg(kwargs.get("use_rslora", None), default=False)
    use_dora = parse_bool_arg(kwargs.get("use_dora", None), default=False)
    init_lora_weights = parse_init_lora_weights_arg(kwargs.get("init_lora_weights", None))

    # Module class override (for LoKr/LoHa reuse of LoRANetwork). Respect the explicit
    # `module_class` parameter (upstream signature); fall back to kwargs for legacy
    # keyword passing, else default to LoRAModule. (Merge fix: the explicit param
    # captured the value, so the old unconditional kwargs.pop always returned the
    # default LoRAModule — silently defeating the non-standard-module guards below.)
    if module_class is None:
        module_class = kwargs.pop("module_class", LoRAModule)
    else:
        kwargs.pop("module_class", None)
    if init_lora_weights != "kaiming" and not issubclass(module_class, LoRAModule):
        raise ValueError(
            f"init_lora_weights={init_lora_weights!r} is only supported by standard LoRA modules, "
            f"got module_class={module_class.__name__}."
        )
    if (rank_pattern or alpha_pattern) and not issubclass(module_class, LoRAModule):
        raise ValueError(
            f"rank_pattern and alpha_pattern are only supported by standard LoRA modules, got module_class={module_class.__name__}."
        )
    if module_kwargs is None:
        module_kwargs = kwargs.pop("module_kwargs", None)
    else:
        kwargs.pop("module_kwargs", None)
    if isinstance(module_kwargs, str):
        module_kwargs = ast.literal_eval(module_kwargs)
    if module_kwargs is not None and not isinstance(module_kwargs, dict):
        raise ValueError(f"module_kwargs must be a dict, got {type(module_kwargs).__name__}")
    enable_conv2d = parse_bool_arg(kwargs.pop("enable_conv2d", None), default=True)

    # too many arguments ( ^ω^)･･･
    network = LoRANetwork(
        target_replace_modules,
        prefix,
        text_encoders,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        conv_lora_dim=conv_dim,
        conv_alpha=conv_alpha,
        module_class=module_class,
        module_kwargs=module_kwargs,
        exclude_patterns=exclude_patterns,
        include_patterns=include_patterns,
        verbose=verbose,
        use_rslora=use_rslora,
        use_dora=use_dora,
        init_lora_weights=init_lora_weights,
        enable_conv2d=enable_conv2d,
        rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern,
    )

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    # loraplus_unet_lr_ratio = kwargs.get("loraplus_unet_lr_ratio", None)
    # loraplus_text_encoder_lr_ratio = kwargs.get("loraplus_text_encoder_lr_ratio", None)
    loraplus_lr_ratio = float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    # loraplus_unet_lr_ratio = float(loraplus_unet_lr_ratio) if loraplus_unet_lr_ratio is not None else None
    # loraplus_text_encoder_lr_ratio = float(loraplus_text_encoder_lr_ratio) if loraplus_text_encoder_lr_ratio is not None else None
    if loraplus_lr_ratio is not None:  # or loraplus_unet_lr_ratio is not None or loraplus_text_encoder_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio)  # , loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio)

    return network


class LoRANetwork(torch.nn.Module):
    # only supports U-Net (DiT), Text Encoders are not supported

    def __init__(
        self,
        target_replace_modules: List[str],
        prefix: str,
        text_encoders: Union[List[CLIPTextModel], CLIPTextModel],
        unet: nn.Module,
        multiplier: float = 1.0,
        lora_dim: int = 4,
        alpha: float = 1,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        conv_lora_dim: Optional[int] = None,
        conv_alpha: Optional[float] = None,
        module_class: Type[object] = LoRAModule,
        module_kwargs: Optional[Dict[str, Any]] = None,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None,
        verbose: Optional[bool] = False,
        use_rslora: bool = False,
        use_dora: bool = False,
        init_lora_weights: str = "kaiming",
        enable_conv2d: bool = True,
        rank_pattern: Optional[Dict[re.Pattern, Union[int, float]]] = None,
        alpha_pattern: Optional[Dict[re.Pattern, Union[int, float]]] = None,
    ) -> None:
        super().__init__()
        self.multiplier = multiplier

        self.lora_dim = lora_dim
        self.alpha = alpha
        self.conv_lora_dim = conv_lora_dim
        self.conv_alpha = conv_alpha
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.target_replace_modules = target_replace_modules
        self.prefix = prefix
        self.module_kwargs = module_kwargs or {}
        self.enable_conv2d = enable_conv2d
        self.init_lora_weights = parse_init_lora_weights_arg(init_lora_weights)
        self.rank_pattern = rank_pattern or {}
        self.alpha_pattern = alpha_pattern or {}
        self._persist_init_lora_weights_metadata = issubclass(module_class, LoRAModule) and modules_dim is None

        # RS-LoRA and DoRA flags with network-level buffers for auto-detection
        self.use_rslora = use_rslora
        self.use_dora = use_dora
        self.register_buffer("use_rslora_flag", torch.tensor(use_rslora, dtype=torch.bool))
        self.register_buffer("use_dora_flag", torch.tensor(use_dora, dtype=torch.bool))

        self.loraplus_lr_ratio = None
        # self.loraplus_unet_lr_ratio = None
        # self.loraplus_text_encoder_lr_ratio = None

        if modules_dim is not None:
            logger.info("create LoRA network from weights")
        else:
            logger.info(f"create LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")
            logger.info(
                f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}"
            )
            # if self.conv_lora_dim is not None:
            #     logger.info(
            #         f"apply LoRA to Conv2d with kernel size (3,3). dim (rank): {self.conv_lora_dim}, alpha: {self.conv_alpha}"
            #     )
        # if train_t5xxl:
        #     logger.info(f"train T5XXL as well")

        # compile regular expression if specified
        exclude_re_patterns = []
        if exclude_patterns is not None:
            for pattern in exclude_patterns:
                try:
                    re_pattern = re.compile(pattern)
                except re.error as e:
                    logger.error(f"Invalid exclude pattern '{pattern}': {e}")
                    continue
                exclude_re_patterns.append(re_pattern)

        include_re_patterns = []
        if include_patterns is not None:
            for pattern in include_patterns:
                try:
                    re_pattern = re.compile(pattern)
                except re.error as e:
                    logger.error(f"Invalid include pattern '{pattern}': {e}")
                    continue
                include_re_patterns.append(re_pattern)

        # create module instances
        def create_modules(
            is_unet: bool,
            pfx: str,
            root_module: torch.nn.Module,
            target_replace_mods: Optional[List[str]] = None,
            filter: Optional[str] = None,
            default_dim: Optional[int] = None,
        ) -> List[LoRAModule]:
            loras = []
            skipped = []
            for name, module in root_module.named_modules():
                if target_replace_mods is None or module.__class__.__name__ in target_replace_mods:
                    if target_replace_mods is None:  # dirty hack for all modules
                        module = root_module  # search all modules

                    for child_name, child_module in module.named_modules():
                        is_linear = child_module.__class__.__name__ == "Linear"
                        is_conv2d = child_module.__class__.__name__ == "Conv2d"
                        is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                        # Skip Conv2d when disabled (e.g. LoKr Linear-only)
                        if is_conv2d and not self.enable_conv2d:
                            conv2d_skipped_count = getattr(self, "_conv2d_skipped_count", 0) + 1
                            self._conv2d_skipped_count = conv2d_skipped_count
                            continue

                        if is_linear or is_conv2d:
                            original_name = (name + "." if name else "") + child_name
                            lora_name = f"{pfx}.{original_name}".replace(".", "_")

                            # exclude/include filter
                            excluded = False
                            for pattern in exclude_re_patterns:
                                if pattern.fullmatch(original_name):
                                    excluded = True
                                    break
                            included = False
                            for pattern in include_re_patterns:
                                if pattern.fullmatch(original_name):
                                    included = True
                                    break
                            if excluded and not included:
                                if verbose:
                                    logger.info(f"exclude: {original_name}")
                                continue

                            # filter by name (not used in the current implementation)
                            if filter is not None and filter not in lora_name:
                                continue

                            dim = None
                            alpha = None

                            if modules_dim is not None:
                                # モジュール指定あり
                                if lora_name in modules_dim:
                                    dim = modules_dim[lora_name]
                                    alpha = modules_alpha.get(lora_name, dim)  # default alpha = dim (LoRA convention)
                            else:
                                # 通常、すべて対象とする
                                if is_linear or is_conv2d_1x1:
                                    dim = default_dim if default_dim is not None else self.lora_dim
                                    alpha = self.alpha
                                elif self.conv_lora_dim is not None:
                                    dim = self.conv_lora_dim
                                    alpha = self.conv_alpha

                                rank_override, rank_pattern_str = _match_pattern_value(self.rank_pattern, original_name)
                                alpha_override, alpha_pattern_str = _match_pattern_value(self.alpha_pattern, original_name)
                                if rank_override is not None:
                                    dim = rank_override
                                    if alpha is None:
                                        alpha = self.alpha
                                if alpha_override is not None and dim is not None:
                                    alpha = alpha_override

                            if dim is None or dim == 0:
                                # skipした情報を出力
                                if is_linear or is_conv2d_1x1 or (self.conv_lora_dim is not None):
                                    skipped.append(lora_name)
                                continue

                            if self.init_lora_weights == "orthogonal" and dim % 2 != 0:
                                if modules_dim is not None:
                                    source = "loaded weights"
                                elif rank_pattern_str is not None:
                                    source = f"rank_pattern {rank_pattern_str!r}"
                                elif alpha_pattern_str is not None:
                                    source = f"alpha_pattern {alpha_pattern_str!r} with default rank"
                                else:
                                    source = "default network_dim/conv_dim"
                                raise ValueError(
                                    f"Module {original_name!r} resolved to rank {dim} via {source}, "
                                    "but init_lora_weights='orthogonal' requires even rank. "
                                    "Adjust the matching pattern or use init_lora_weights=kaiming."
                                )

                            lora = module_class(
                                lora_name,
                                child_module,
                                self.multiplier,
                                dim,
                                alpha,
                                dropout=dropout,
                                rank_dropout=rank_dropout,
                                module_dropout=module_dropout,
                                use_rslora=self.use_rslora,
                                use_dora=self.use_dora,
                                init_lora_weights=self.init_lora_weights,
                                **self.module_kwargs,
                            )
                            loras.append(lora)

                if target_replace_mods is None:
                    break  # all modules are searched
            return loras, skipped

        # # create LoRA for text encoder
        # # it is redundant to create LoRA modules even if they are not used

        self.text_encoder_loras: List[Union[LoRAModule, LoRAInfModule]] = []
        # skipped_te = []
        # for i, text_encoder in enumerate(text_encoders):
        #     index = i
        #     if not train_t5xxl and index > 0:  # 0: CLIP, 1: T5XXL, so we skip T5XXL if train_t5xxl is False
        #         break
        #     logger.info(f"create LoRA for Text Encoder {index+1}:")
        #     text_encoder_loras, skipped = create_modules(False, index, text_encoder, LoRANetwork.TEXT_ENCODER_TARGET_REPLACE_MODULE)
        #     logger.info(f"create LoRA for Text Encoder {index+1}: {len(text_encoder_loras)} modules.")
        #     self.text_encoder_loras.extend(text_encoder_loras)
        #     skipped_te += skipped

        # create LoRA for U-Net
        self.unet_loras: List[Union[LoRAModule, LoRAInfModule]]
        self.unet_loras, skipped_un = create_modules(True, prefix, unet, target_replace_modules)

        if hasattr(self, "_conv2d_skipped_count") and self._conv2d_skipped_count > 0:
            logger.warning(f"Skipped {self._conv2d_skipped_count} Conv2d modules (enable_conv2d=False)")

        self._orthogonal_conv2d_fallback_count = sum(
            1
            for lora in self.text_encoder_loras + self.unet_loras
            if getattr(lora, "orthogonal_init_fallback_reason", None) == "Conv2d"
        )
        if self._orthogonal_conv2d_fallback_count > 0:
            logger.warning(
                "Orthogonal init requested, but "
                f"{self._orthogonal_conv2d_fallback_count} Conv2d LoRA modules fell back to kaiming init."
            )

        logger.info(f"create LoRA for U-Net/DiT: {len(self.unet_loras)} modules.")
        if verbose:
            for lora in self.unet_loras:
                logger.info(f"\t{lora.lora_name:50} {lora.lora_dim}, {lora.alpha}")

        skipped = skipped_un
        if verbose and len(skipped) > 0:
            logger.warning(
                f"because dim (rank) is 0, {len(skipped)} LoRA modules are skipped / dim (rank)が0の為、次の{len(skipped)}個のLoRAモジュールはスキップされます:"
            )
            for name in skipped:
                logger.info(f"\t{name}")

        # DoRA summary: count modules where DoRA was disabled
        if use_dora:
            dora_enabled = 0
            dora_disabled_counts = {}  # reason -> count
            for lora in self.text_encoder_loras + self.unet_loras:
                if lora.use_dora:
                    dora_enabled += 1
                elif hasattr(lora, "dora_disabled_reason") and lora.dora_disabled_reason:
                    reason = lora.dora_disabled_reason
                    dora_disabled_counts[reason] = dora_disabled_counts.get(reason, 0) + 1
            if dora_disabled_counts:
                disabled_summary = ", ".join(f"{count} {reason}" for reason, count in dora_disabled_counts.items())
                logger.info(f"DoRA enabled on {dora_enabled} modules, disabled on: {disabled_summary}")
            else:
                logger.info(f"DoRA enabled on all {dora_enabled} modules")

        # assertion
        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

    def prepare_network(self, args):
        """
        called after the network is created
        """
        pass

    def set_multiplier(self, multiplier):
        self.multiplier = multiplier
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.multiplier = self.multiplier

    def set_enabled(self, is_enabled):
        for lora in self.text_encoder_loras + self.unet_loras:
            lora.enabled = is_enabled

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu", weights_only=True)

        # Check for RS-LoRA/DoRA flag mismatches before loading
        loaded_rslora = weights_sd.get("use_rslora_flag", torch.tensor(False)).item() if "use_rslora_flag" in weights_sd else False

        # DoRA detection: check flag first, fallback to scanning for dora_layer.weight keys
        loaded_dora = False
        if "use_dora_flag" in weights_sd:
            loaded_dora = weights_sd["use_dora_flag"].item()
        else:
            # Fallback: scan for dora_layer.weight keys (for older/external weights)
            for key in weights_sd.keys():
                if "dora_layer.weight" in key:
                    loaded_dora = True
                    break

        if loaded_rslora != self.use_rslora:
            # Build error message with concrete fix options
            err_msg = (
                f"RS-LoRA flag mismatch: weights have use_rslora={loaded_rslora}, but network was created with use_rslora={self.use_rslora}. "
                f"This will produce incorrect scaling (alpha/r vs alpha/sqrt(r)). "
                f"Fix: recreate the network with use_rslora={loaded_rslora}, or re-save the weights with use_rslora_flag={self.use_rslora}."
            )

            # If flag is missing (loaded_rslora=False) but network expects RS-LoRA,
            # check for suspicious alphas that suggest the weights were actually RS-LoRA trained
            if not loaded_rslora and self.use_rslora and "use_rslora_flag" not in weights_sd:
                # Check if many alpha values are close to sqrt(dim) - suggests RS-LoRA with alpha=0
                # Note: only checks {name}.lora_down.weight, split-dims weights won't be counted
                suspicious_count = 0
                total_count = 0
                for key, value in weights_sd.items():
                    if key.endswith(".alpha") and "." in key:
                        lora_name = key.rsplit(".", 1)[0]
                        down_key = f"{lora_name}.lora_down.weight"
                        if down_key in weights_sd:
                            dim = weights_sd[down_key].shape[0]
                            alpha_val = value.item() if hasattr(value, "item") else float(value)
                            total_count += 1
                            if abs(alpha_val - math.sqrt(dim)) < 0.01:
                                suspicious_count += 1
                if total_count > 0 and suspicious_count / total_count > 0.5:
                    err_msg += (
                        f" HINT: {suspicious_count}/{total_count} alpha values equal sqrt(dim), which suggests "
                        f"these weights may have been trained with RS-LoRA (alpha=0). The use_rslora_flag buffer "
                        f"may have been stripped. Consider using use_rslora=True."
                    )

            raise ValueError(err_msg)
        # DoRA mismatch handling is asymmetric:
        # - use_dora=True but weights lack DoRA: hard error (uninitialized magnitude = wrong results)
        # - use_dora=False but weights have DoRA: warn (intentionally ignoring DoRA is reasonable)
        if self.use_dora and not loaded_dora:
            raise ValueError(
                "DoRA flag mismatch: network expects DoRA (use_dora=True), but weights have no DoRA magnitude vectors. "
                "This will produce incorrect results with uninitialized magnitudes. "
                "Either recreate the network with use_dora=False, or use DoRA-trained weights."
            )
        elif not self.use_dora and loaded_dora:
            logger.warning(
                "DoRA flag mismatch: weights contain DoRA magnitude vectors, but network was created with use_dora=False. "
                "DoRA magnitudes will be ignored and weights will be treated as standard LoRA. "
                "If this is unintentional, recreate the network with use_dora=True."
            )

        # Sanitize state dict to avoid propagating contradictory metadata buffers.
        # If we intentionally ignore DoRA, do not allow `use_dora_flag=True` to flip our buffer.
        if not self.use_dora:
            weights_sd.pop("use_dora_flag", None)
            weights_sd["use_dora_flag"] = torch.tensor(False, dtype=torch.bool)

        # Fail fast on partial DoRA checkpoints: if a module has LoRA weights, it must have a DoRA magnitude too.
        if self.use_dora and loaded_dora:
            # Prefer the in-memory LoRA module objects (works even before apply_to()).
            name_to_lora = {m.lora_name: m for m in (self.text_encoder_loras + self.unet_loras)}
            prefixes: set[str] = set()
            for k in weights_sd.keys():
                if k.endswith(".lora_down.weight") or k.endswith(".lora_up.weight"):
                    prefixes.add(k.rsplit(".", 2)[0])
            missing_mag = []
            for p in sorted(prefixes):
                mod = getattr(self, p, None) or name_to_lora.get(p)
                if mod is None or not getattr(mod, "use_dora", False):
                    continue
                mag_key = f"{p}.dora_layer.weight"
                if mag_key not in weights_sd:
                    missing_mag.append(mag_key)
            if missing_mag:
                preview = ", ".join(missing_mag[:5])
                raise ValueError(
                    f"DoRA checkpoint appears partial: missing {len(missing_mag)} dora_layer.weight tensors "
                    f"for modules that have LoRA weights. Example(s): {preview}. "
                    f"This would silently use uninitialized magnitudes (all-ones) and produce incorrect results."
                )

        info = self.load_state_dict(weights_sd, False)
        return info

    def apply_to(
        self,
        text_encoders: Optional[nn.Module],
        unet: Optional[nn.Module],
        apply_text_encoder: bool = True,
        apply_unet: bool = True,
    ):
        if apply_text_encoder:
            logger.info(f"enable LoRA for text encoder: {len(self.text_encoder_loras)} modules")
        else:
            self.text_encoder_loras = []

        if apply_unet:
            logger.info(f"enable LoRA for U-Net: {len(self.unet_loras)} modules")
        else:
            self.unet_loras = []

        if len(self.text_encoder_loras) == 0 and len(self.unet_loras) == 0:
            logger.error(
                "No LoRA modules. Please check `--network_module` and `--network_args`"
                " / LoRAモジュールがありません。`--network_module`と`--network_args`を確認してください"
            )
            raise RuntimeError("No LoRA modules found")

        for lora in self.text_encoder_loras + self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    # マージできるかどうかを返す
    def is_mergeable(self):
        return True

    # TODO refactor to common function with apply_to
    def merge_to(self, text_encoders, unet, weights_sd, dtype=None, device=None, non_blocking=False, safe_merge: bool = False):
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=2) as executor:  # 2 workers is enough
            futures = []
            for lora in self.text_encoder_loras + self.unet_loras:
                sd_for_lora = {}
                for key in weights_sd.keys():
                    if key.startswith(lora.lora_name):
                        sd_for_lora[key[len(lora.lora_name) + 1 :]] = weights_sd[key]
                if len(sd_for_lora) == 0:
                    logger.info(f"no weight for {lora.lora_name}")
                    continue

                # lora.merge_to(sd_for_lora, dtype, device)
                futures.append(executor.submit(lora.merge_to, sd_for_lora, dtype, device, non_blocking, safe_merge))

        for future in futures:
            future.result()

        # logger.info(f"weights are merged")

    def set_loraplus_lr_ratio(self, loraplus_lr_ratio):  # , loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio):
        self.loraplus_lr_ratio = loraplus_lr_ratio

        logger.info(f"LoRA+ UNet LR Ratio: {self.loraplus_lr_ratio}")
        # logger.info(f"LoRA+ Text Encoder LR Ratio: {self.loraplus_text_encoder_lr_ratio or self.loraplus_lr_ratio}")

    def prepare_optimizer_params(self, unet_lr: float = 1e-4, **kwargs):
        self.requires_grad_(True)

        all_params = []
        lr_descriptions = []

        def assemble_params(loras, lr, loraplus_ratio):
            param_groups = {"lora": {}, "plus": {}}
            for lora in loras:
                for name, param in lora.named_parameters():
                    if loraplus_ratio is not None and "lora_up" in name:
                        param_groups["plus"][f"{lora.lora_name}.{name}"] = param
                    else:
                        param_groups["lora"][f"{lora.lora_name}.{name}"] = param

            if loraplus_ratio is not None and len(param_groups["plus"]) == 0:
                logger.warning("LoRA+ is not effective for this network type (no 'lora_up' parameters found)")

            params = []
            descriptions = []
            for key in param_groups.keys():
                param_data = {"params": param_groups[key].values()}

                if len(param_data["params"]) == 0:
                    continue

                if lr is not None:
                    if key == "plus":
                        param_data["lr"] = lr * loraplus_ratio
                    else:
                        param_data["lr"] = lr

                if param_data.get("lr", None) == 0 or param_data.get("lr", None) is None:
                    logger.info("NO LR skipping!")
                    continue

                params.append(param_data)
                descriptions.append("plus" if key == "plus" else "")

            return params, descriptions

        if self.unet_loras:
            params, descriptions = assemble_params(self.unet_loras, unet_lr, self.loraplus_lr_ratio)
            all_params.extend(params)
            lr_descriptions.extend(["unet" + (" " + d if d else "") for d in descriptions])

            if self.loraplus_lr_ratio is not None and "plus" not in descriptions:
                logger.warning(
                    "LoRA+ has no active plus param group (check loraplus_lr_ratio and lr settings, "
                    "or this network type may lack 'lora_up' parameters)"
                )

        return all_params, lr_descriptions

    def enable_gradient_checkpointing(self):
        # not supported
        pass

    def prepare_grad_etc(self, unet):
        self.requires_grad_(True)

    def on_epoch_start(self, unet):
        self.train()

    def on_step_start(self):
        pass

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, dtype, metadata):
        if metadata is not None and len(metadata) == 0:
            metadata = None

        state_dict = self.state_dict()

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file
            from musubi_tuner.utils import model_utils

            # Precalculate model hashes to save time on indexing
            if metadata is None:
                metadata = {}
            if self._persist_init_lora_weights_metadata:
                metadata.setdefault("ss_init_lora_weights", self.init_lora_weights)
                if self.rank_pattern:
                    metadata["ss_rank_pattern"] = _pattern_metadata_json(self.rank_pattern)
                if self.alpha_pattern:
                    metadata["ss_alpha_pattern"] = _pattern_metadata_json(self.alpha_pattern)
                if self.rank_pattern or self.alpha_pattern:
                    metadata["ss_rank_pattern_match_semantics"] = RANK_PATTERN_MATCH_SEMANTICS
            model_hash, legacy_hash = model_utils.precalculate_safetensors_hashes(state_dict, metadata)
            metadata["sshs_model_hash"] = model_hash
            metadata["sshs_legacy_hash"] = legacy_hash

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

    def backup_weights(self):
        # 重みのバックアップを行う
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not hasattr(org_module, "_lora_org_weight"):
                sd = org_module.state_dict()
                org_module._lora_org_weight = sd["weight"].detach().clone()
                org_module._lora_restored = True

    def restore_weights(self):
        # 重みのリストアを行う
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            if not org_module._lora_restored:
                sd = org_module.state_dict()
                sd["weight"] = org_module._lora_org_weight
                org_module.load_state_dict(sd)
                org_module._lora_restored = True

    def pre_calculation(self):
        # 事前計算を行う
        loras: List[LoRAInfModule] = self.text_encoder_loras + self.unet_loras
        for lora in loras:
            org_module = lora.org_module_ref[0]
            sd = org_module.state_dict()

            org_weight = sd["weight"]
            lora_weight = lora.get_weight().to(org_weight.device, dtype=org_weight.dtype)
            sd["weight"] = org_weight + lora_weight
            assert sd["weight"].shape == org_weight.shape
            org_module.load_state_dict(sd)

            org_module._lora_restored = False
            lora.enabled = False

    def apply_max_norm_regularization(self, max_norm_value, device):
        downkeys = []
        upkeys = []
        alphakeys = []
        norms = []
        keys_scaled = 0

        state_dict = self.state_dict()

        # guard: only supported for LoRA (lora_down/lora_up parameterization)
        if not any("lora_down" in k and "weight" in k for k in state_dict.keys()):
            logger.warning("max_norm_regularization is only supported for LoRA")
            return 0, 0.0, 0.0

        for key in state_dict.keys():
            if "lora_down" in key and "weight" in key:
                # Skip split_dims keys (lora_down.0.weight, etc.) - not supported
                if re.search(r"lora_down\.\d+\.weight", key):
                    continue
                downkeys.append(key)
                upkeys.append(key.replace("lora_down", "lora_up"))
                alphakeys.append(key.replace("lora_down.weight", "alpha"))

        # Guard: only supported for LoRA (lora_down/lora_up parameterization)
        if not downkeys:
            logger.warning("max_norm_regularization is only supported for LoRA (no lora_down keys found)")
            return 0, 0.0, 0.0

        for i in range(len(downkeys)):
            down = state_dict[downkeys[i]].to(device)
            up = state_dict[upkeys[i]].to(device)
            alpha = state_dict[alphakeys[i]].to(device)
            dim = down.shape[0]

            # RS-LoRA scaling
            if self.use_rslora:
                scale = alpha / math.sqrt(dim)
            else:
                scale = alpha / dim

            if up.shape[2:] == (1, 1) and down.shape[2:] == (1, 1):
                updown = (up.squeeze(2).squeeze(2) @ down.squeeze(2).squeeze(2)).unsqueeze(2).unsqueeze(3)
            elif up.shape[2:] == (3, 3) or down.shape[2:] == (3, 3):
                updown = torch.nn.functional.conv2d(down.permute(1, 0, 2, 3), up).permute(1, 0, 2, 3)
            else:
                updown = up @ down

            updown *= scale

            norm = updown.norm().clamp(min=max_norm_value / 2)
            desired = torch.clamp(norm, max=max_norm_value)
            ratio = (desired / norm).item()  # Python float avoids CPU/CUDA device mismatch
            sqrt_ratio = ratio**0.5
            if ratio != 1:
                keys_scaled += 1
                state_dict[upkeys[i]] *= sqrt_ratio
                state_dict[downkeys[i]] *= sqrt_ratio
            scalednorm = updown.norm().item() * ratio
            norms.append(scalednorm)

        if not norms:
            return keys_scaled, 0.0, 0.0
        return keys_scaled, sum(norms) / len(norms), max(norms)


def create_arch_network_from_weights(
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    **kwargs,
) -> LoRANetwork:
    return create_network_from_weights(
        HUNYUAN_TARGET_REPLACE_MODULES, multiplier, weights_sd, text_encoders, unet, for_inference, **kwargs
    )


# Create network from weights for inference, weights are not loaded here (because can be merged)
def create_network_from_weights(
    target_replace_modules: List[str],
    multiplier: float,
    weights_sd: Dict[str, torch.Tensor],
    text_encoders: Optional[List[nn.Module]] = None,
    unet: Optional[nn.Module] = None,
    for_inference: bool = False,
    module_class: Optional[Type[object]] = None,
    module_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs,
) -> LoRANetwork:
    # get dim/alpha mapping
    modules_dim = {}
    modules_alpha = {}

    # Auto-detect RS-LoRA from network-level buffer
    use_rslora = False
    if "use_rslora_flag" in weights_sd:
        use_rslora = weights_sd["use_rslora_flag"].item()

    # Override with explicit kwarg if provided
    if "use_rslora" in kwargs:
        use_rslora = parse_bool_arg(kwargs.get("use_rslora"), default=use_rslora)

    # Auto-detect DoRA from network-level buffer
    use_dora = False
    if "use_dora_flag" in weights_sd:
        use_dora = weights_sd["use_dora_flag"].item()
    else:
        # Fallback: scan for dora_layer.weight keys (for older/external weights)
        for key in weights_sd.keys():
            if "dora_layer.weight" in key:
                use_dora = True
                break

    # Override with explicit kwarg if provided
    if "use_dora" in kwargs:
        use_dora = parse_bool_arg(kwargs.get("use_dora"), default=use_dora)

    for key, value in weights_sd.items():
        if "." not in key:
            continue

        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            # Accept all lora_down keys including split_dims (lora_down.0.weight, etc.)
            # to maintain backward compatibility with existing weights
            dim = value.shape[0]
            modules_dim[lora_name] = dim
        elif "lokr_w2_a" in key:
            # LoKr low-rank mode: dim is the inner dimension of w2_a
            dim = value.shape[1]
            modules_dim[lora_name] = dim
        elif "lokr_w2" in key and "lokr_w2_a" not in key and "lokr_w2_b" not in key:
            # LoKr full-matrix mode: dim is max of w2 shape
            dim = max(value.shape)
            modules_dim[lora_name] = dim

    # Module class / kwargs override (for LoKr/LoHa reuse of LoRANetwork). Respect the explicit
    # parameters (upstream signature); fall back to kwargs for legacy keyword passing. (Merge fix:
    # the old unconditional `kwargs.pop("module_kwargs", None)` silently discarded a caller-supplied
    # `module_kwargs` bound to the named param — e.g. LoKr's {"factor": ...} from
    # create_arch_network_from_weights — defeating _resolve_factor and forcing factor=-1 (automatic
    # factorization) on reload, mismatching saved lokr_w1/lokr_w2 shapes. Mirrors the guard in
    # create_network() above.)
    if module_class is None:
        module_class = kwargs.pop("module_class", LoRAInfModule if for_inference else LoRAModule)
    else:
        kwargs.pop("module_class", None)
    if module_kwargs is None:
        module_kwargs = kwargs.pop("module_kwargs", None)
    else:
        kwargs.pop("module_kwargs", None)
    if isinstance(module_kwargs, str):
        module_kwargs = ast.literal_eval(module_kwargs)
    if module_kwargs is not None and not isinstance(module_kwargs, dict):
        raise ValueError(f"module_kwargs must be a dict, got {type(module_kwargs).__name__}")
    enable_conv2d = parse_bool_arg(kwargs.pop("enable_conv2d", None), default=True)

    network = LoRANetwork(
        target_replace_modules,
        "lora_unet",
        text_encoders,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
        module_kwargs=module_kwargs,
        use_rslora=use_rslora,
        use_dora=use_dora,
        enable_conv2d=enable_conv2d,
    )
    return network
