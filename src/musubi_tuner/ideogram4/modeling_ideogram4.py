"""Ideogram4 transformer backbone.

The transformer consumes Qwen3-VL embeddings and flow-matching noise tokens to
produce velocity predictions on image latents.
"""

from __future__ import annotations

import math
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from musubi_tuner.ideogram4.constants import LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR, QWEN3_VL_ACTIVATION_LAYERS
from musubi_tuner.utils.model_utils import create_cpu_offloading_wrapper

# LOAD-BEARING IMPORT — do NOT delete as "unused" (that is why it carries noqa, not an alias use).
# Importing modules.attention runs a process-wide guard that disables the cuDNN SDPA backend on
# Blackwell GPUs (sm_10x/sm_12x), where cuDNN's fused-attention backward produces NaN/crashes during
# training. Ideogram4Attention below calls F.scaled_dot_product_attention directly with no per-call
# backend selection, so this guard MUST already be active at first forward. Verified firing on RTX
# 5090 (prints "Blackwell GPU detected … disabling cuDNN SDPA backend"). Reuse-by-import is
# deliberate (one guard, not a duplicate) — see docs/plans/2026-06-07-ideogram4-fork-review.md §F4.
from musubi_tuner.modules import attention as _attention_guard  # noqa: F401


@dataclass
class Ideogram4Config:
    emb_dim: int = 4608
    num_layers: int = 34
    num_heads: int = 18
    intermediate_size: int = 12288
    adanln_dim: int = 512

    # Latent dimension after patchification: ae_channels (32) * patch_size**2 (4) = 128.
    in_channels: int = 128

    # Hidden size of Qwen3-VL-8B-Instruct multiplied by the number of layers we extract
    # Qwen3-VL hidden size = 4096
    llm_features_dim: int = 4096 * len(QWEN3_VL_ACTIVATION_LAYERS)

    rope_theta: int = 5_000_000
    mrope_section: tuple[int, ...] = (24, 20, 20)

    norm_eps: float = 1e-5


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: (B, num_heads, L, head_dim); cos/sin: (B, L, head_dim).
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    q_embed = (q * cos) + (_rotate_half(q) * sin)
    k_embed = (k * cos) + (_rotate_half(k) * sin)
    return q_embed, k_embed


class Ideogram4MRoPE(nn.Module):
    inv_freq: torch.Tensor

    def __init__(
        self,
        head_dim: int,
        base: int,
        mrope_section: tuple[int, ...],
    ) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.base = base
        self.mrope_section = tuple(mrope_section)
        # inv_freq must stay float32 for RoPE precision (the forward upcasts it). It is a buffer so it
        # follows the model's device, but a model-wide .to(bf16) downcasts it and the upcast cannot
        # recover the lost mantissa — call reset_inv_freq() after any such cast. See the loader.
        self.register_buffer("inv_freq", self._compute_inv_freq(torch.device("cpu")), persistent=False)

    def _compute_inv_freq(self, device: torch.device) -> torch.Tensor:
        return 1.0 / (self.base ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=device) / self.head_dim))

    def reset_inv_freq(self) -> None:
        """Rebuild inv_freq in float32 on its current device (recovers precision after a bf16 model cast)."""
        self.register_buffer("inv_freq", self._compute_inv_freq(self.inv_freq.device), persistent=False)

    def _apply(self, fn, *args, **kwargs):
        # Self-heal the float32 invariant: .to()/.cuda()/.bfloat16()/accelerate device+dtype moves all
        # route through _apply and would downcast inv_freq. Rebuild it float32 on the post-move device so
        # the invariant holds without any caller remembering to repair it. Only rebuild when the dtype
        # actually drifted (a pure device move already preserved float32); skip on meta (rebuilt on realize).
        module = super()._apply(fn, *args, **kwargs)
        if module.inv_freq.dtype != torch.float32 and module.inv_freq.device.type != "meta":
            module.reset_inv_freq()
        return module

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: (B, L, 3) of int.
        assert position_ids.ndim == 3 and position_ids.shape[-1] == 3
        batch_size, seq_len, _ = position_ids.shape

        # The MRoPE frequencies must be computed in real fp32. Image position ids start at
        # IMAGE_POSITION_OFFSET (2**16), where bf16 has a spacing of 512, so under an active autocast
        # context the `inv_freq @ pos` matmul would be downcast to bf16 and every image position would
        # collapse to a single value -- wiping out the spatial RoPE phase (a flat checkerboard). The
        # fp32 `inv_freq` buffer and the explicit `.to(torch.float32)` below do NOT protect against
        # this, because autocast overrides matmul's compute dtype regardless of input dtype; we must
        # force autocast off for the whole computation. This is a no-op when no outer autocast is
        # active (blissful's DiT forward currently runs autocast-off), so it never changes existing
        # results -- it is correctness hardening for any future mixed-precision DiT path. Ported from
        # upstream musubi-tuner #975.
        device_type = position_ids.device.type
        autocast_disabled = (
            torch.autocast(device_type=device_type, enabled=False)
            if device_type in ("cuda", "cpu", "xpu", "hpu")
            else nullcontext()
        )
        with autocast_disabled:
            # (3, B, inv_freq_size, L)
            pos = position_ids.permute(2, 0, 1).to(dtype=torch.float32)  # type: ignore[arg-type]
            inv_freq = self.inv_freq.to(dtype=torch.float32)[None, None, :, None].expand(3, batch_size, -1, 1)  # type: ignore[index]
            freqs = inv_freq @ pos.unsqueeze(2)
            freqs = freqs.transpose(2, 3)  # (3, B, L, inv_freq_size)

            # interleaved mrope: pull H freqs into idx 1 mod 3, W freqs into idx 2 mod 3.
            freqs_t = freqs[0].clone()
            for axis, offset in ((1, 1), (2, 2)):
                length = self.mrope_section[axis] * 3
                idx = torch.arange(offset, length, 3, device=freqs_t.device)
                freqs_t[..., idx] = freqs[axis][..., idx]

            emb = torch.cat((freqs_t, freqs_t), dim=-1)
            return emb.cos(), emb.sin()


class Ideogram4RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, self.weight.shape, self.weight, self.eps)


class Ideogram4Attention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, eps: float = 1e-5) -> None:
        super().__init__()
        assert hidden_size % num_heads == 0
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.qkv = nn.Linear(hidden_size, hidden_size * 3, bias=False)
        self.norm_q = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.norm_k = Ideogram4RMSNorm(self.head_dim, eps=eps)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q = self.norm_q(q)
        k = self.norm_k(k)

        # SDPA expects (B, num_heads, L, head_dim).
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = _apply_rotary_pos_emb(q, k, cos, sin)

        # attn_mask is the block-diagonal (B,1,L,L) mask built ONCE in the root forward, or None for a
        # single-sample sequence where it is all-True. Passing None lets SDPA select the flash backend
        # (flash rejects a non-null mask) -- numerically a no-op for batch=1 (no padding), a real speedup.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        return self.o(out)


class Ideogram4MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class Ideogram4TransformerBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_heads: int,
        norm_eps: float,
        adanln_dim: int,
    ) -> None:
        super().__init__()
        self.attention = Ideogram4Attention(hidden_size, num_heads, eps=1e-5)
        self.feed_forward = Ideogram4MLP(hidden_size, intermediate_size)

        self.attention_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm1 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.attention_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)
        self.ffn_norm2 = Ideogram4RMSNorm(hidden_size, eps=norm_eps)

        self.adaln_modulation = nn.Linear(adanln_dim, 4 * hidden_size, bias=True)

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def enable_gradient_checkpointing(self, activation_cpu_offloading: bool = False) -> None:
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = activation_cpu_offloading

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False

    def forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        adaln_input: torch.Tensor,
    ) -> torch.Tensor:
        # Block-internal gradient checkpointing. Gate on torch.is_grad_enabled() (NOT self.training): this
        # recomputes the block in backward to save activation VRAM, and must NOT fire under any no_grad() path
        # (sampling, debug probes). use_reentrant=False is mandatory — the base DiT is frozen and only the LoRA
        # adapters inside the nn.Linear forwards are trainable; the reentrant path would drop their gradients.
        if torch.is_grad_enabled() and self.gradient_checkpointing:
            fn = self._forward
            if self.activation_cpu_offloading:
                fn = create_cpu_offloading_wrapper(fn, self.feed_forward.w1.weight.device)
            return checkpoint(fn, x, attn_mask, cos, sin, adaln_input, use_reentrant=False)
        return self._forward(x, attn_mask, cos, sin, adaln_input)

    def _forward(
        self,
        x: torch.Tensor,
        attn_mask: torch.Tensor | None,
        cos: torch.Tensor,
        sin: torch.Tensor,
        adaln_input: torch.Tensor,
    ) -> torch.Tensor:
        mod = self.adaln_modulation(adaln_input)
        scale_msa, gate_msa, scale_mlp, gate_mlp = mod.chunk(4, dim=-1)
        gate_msa = torch.tanh(gate_msa)
        gate_mlp = torch.tanh(gate_mlp)
        scale_msa = 1.0 + scale_msa
        scale_mlp = 1.0 + scale_mlp

        attn_out = self.attention(
            self.attention_norm1(x) * scale_msa,
            attn_mask=attn_mask,
            cos=cos,
            sin=sin,
        )
        x = x + gate_msa * self.attention_norm2(attn_out)
        x = x + gate_mlp * self.ffn_norm2(self.feed_forward(self.ffn_norm1(x) * scale_mlp))
        return x


def _sinusoidal_embedding(t: torch.Tensor, dim: int, scale: float = 1e4) -> torch.Tensor:
    t = t.to(torch.float32)
    half = dim // 2
    freq = math.log(scale) / (half - 1)
    freq = torch.exp(torch.arange(half, dtype=torch.float32, device=t.device) * -freq)  # type: ignore[assignment]
    emb = t.unsqueeze(-1) * freq
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class Ideogram4EmbedScalar(nn.Module):
    def __init__(self, dim: int, input_range: tuple[float, float]) -> None:
        super().__init__()
        self.dim = dim
        self.range_min, self.range_max = input_range
        assert self.range_max > self.range_min
        self.mlp_in = nn.Linear(dim, dim, bias=True)
        self.mlp_out = nn.Linear(dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is shape (..., 1) or (...,) holding a scalar per token.
        x = x.to(torch.float32)
        scaled = 1e4 * (x - self.range_min) / (self.range_max - self.range_min)
        emb = _sinusoidal_embedding(scaled, self.dim)
        emb = emb.to(getattr(self.mlp_in, "compute_dtype", None) or self.mlp_in.weight.dtype)
        emb = F.silu(self.mlp_in(emb))
        return self.mlp_out(emb)


class Ideogram4FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, out_channels: int, adanln_dim: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, eps=1e-6, elementwise_affine=False)
        self.linear = nn.Linear(hidden_size, out_channels, bias=True)
        self.adaln_modulation = nn.Linear(adanln_dim, hidden_size, bias=True)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.adaln_modulation(F.silu(c))
        return self.linear(self.norm_final(x) * scale)


class Ideogram4Transformer(nn.Module):
    """Ideogram 4 flow-matching transformer."""

    def __init__(self, config: Ideogram4Config) -> None:
        super().__init__()
        self.config = config

        head_dim = config.emb_dim // config.num_heads

        self.input_proj = nn.Linear(config.in_channels, config.emb_dim, bias=True)
        self.llm_cond_norm = Ideogram4RMSNorm(config.llm_features_dim, eps=1e-6)
        self.llm_cond_proj = nn.Linear(config.llm_features_dim, config.emb_dim, bias=True)
        self.t_embedding = Ideogram4EmbedScalar(config.emb_dim, input_range=(0.0, 1.0))
        self.adaln_proj = nn.Linear(config.emb_dim, config.adanln_dim, bias=True)

        self.embed_image_indicator = nn.Embedding(2, config.emb_dim)

        self.rotary_emb = Ideogram4MRoPE(
            head_dim=head_dim,
            base=config.rope_theta,
            mrope_section=config.mrope_section,
        )

        self.layers = nn.ModuleList(
            [
                Ideogram4TransformerBlock(
                    hidden_size=config.emb_dim,
                    intermediate_size=config.intermediate_size,
                    num_heads=config.num_heads,
                    norm_eps=config.norm_eps,
                    adanln_dim=config.adanln_dim,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_layer = Ideogram4FinalLayer(
            hidden_size=config.emb_dim,
            out_channels=config.in_channels,
            adanln_dim=config.adanln_dim,
        )

        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        # Timestep conditioning precision. Default False = legacy: cast t to the (bf16) compute dtype in
        # forward(); True = keep t in fp32 (the corrected regime). See forward() for why this matters.
        # Set from --ideogram4_fp32_timestep by the trainer/generator after load (before any wrap/compile).
        self.fp32_timestep = False

    def enable_gradient_checkpointing(self, cpu_offload: bool = False) -> None:
        # Base trainer calls this with args.gradient_checkpointing_cpu_offload (trainer_base.py:1661). The
        # Ideogram trainer rejects --gradient_checkpointing_cpu_offload for now, so cpu_offload is False in
        # practice; the per-block wrapper is wired for when a dedicated CPU-offload backward test lands.
        self.gradient_checkpointing = True
        self.activation_cpu_offloading = cpu_offload
        for block in self.layers:
            block.enable_gradient_checkpointing(activation_cpu_offloading=cpu_offload)

    def disable_gradient_checkpointing(self) -> None:
        self.gradient_checkpointing = False
        self.activation_cpu_offloading = False
        for block in self.layers:
            block.disable_gradient_checkpointing()

    def switch_block_swap_for_inference(self) -> None:
        # No-op: Ideogram rejects --blocks_to_swap (no block-swap hooks yet), but the base sampler calls these
        # UNCONDITIONALLY around sample generation (trainer_base.py:868/911). Provide them so sampling-during-
        # training doesn't AttributeError. These become the real implementations when Slice 2 (block swap) lands.
        pass

    def switch_block_swap_for_training(self) -> None:
        pass

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    def forward(
        self,
        *,
        llm_features: torch.Tensor,
        x: torch.Tensor,
        t: torch.Tensor,
        position_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        indicator: torch.Tensor,
    ) -> torch.Tensor:
        """Velocity prediction.

        Args:
            llm_features: (B, L, llm_features_dim) Qwen3-VL conditioning features.
            x: (B, L, in_channels) noise tokens.
            t: (B,) or (B, L) flow-matching time in [0, 1].
            position_ids: (B, L, 3) (t, h, w) positions for MRoPE.
            segment_ids: (B, L) per-position validity partition from build_ideogram4_conditioning
                (1 = real text/image token, SEQUENCE_PADDING_INDICATOR = left-pad). CONTRACT: one sample
                per batch row, so a batch_size==1 row is always unpadded/uniform -- which the attention
                mask-elision below relies on. Single-row packing or fixed-length padding would violate
                this and must revisit the batch_size==1 branch.
            indicator: (B, L) per-token role: LLM_TOKEN_INDICATOR or OUTPUT_IMAGE_INDICATOR.

        Returns:
            (B, L, in_channels) velocity prediction in float32. Only the positions
            with ``indicator == OUTPUT_IMAGE_INDICATOR`` are meaningful.
        """
        batch_size, seq_len, in_channels = x.shape
        assert in_channels == self.config.in_channels

        param_dtype = getattr(self.input_proj, "compute_dtype", None) or self.input_proj.weight.dtype
        x = x.to(param_dtype)
        # t feeds Ideogram4EmbedScalar, which re-upcasts to fp32 (line ~294) and applies a 1e4 sinusoidal
        # scale that amplifies any rounding -- so casting t to the (bf16) param dtype here is a lossy
        # round-trip with no compute benefit (timestep-embedding cosine ~0.79 vs fp32). EmbedScalar casts
        # its OUTPUT back to compute dtype regardless, so this flag changes ONLY the sinusoidal precision;
        # no downstream tensor changes dtype. Default (legacy) keeps the bf16 cast -- the regime every
        # pre-2026-06 Ideogram adapter was trained AND sampled under; --ideogram4_fp32_timestep opts into
        # the corrected fp32 conditioning. See docs/ideogram4.md.
        t = t.to(torch.float32) if self.fp32_timestep else t.to(param_dtype)
        llm_features = llm_features.to(param_dtype)

        indicator = indicator.to(torch.long)
        llm_token_mask = (indicator == LLM_TOKEN_INDICATOR).to(x.dtype).unsqueeze(-1)
        output_image_mask = (indicator == OUTPUT_IMAGE_INDICATOR).to(x.dtype).unsqueeze(-1)

        llm_features = llm_features * llm_token_mask
        x = x * output_image_mask

        x = self.input_proj(x) * output_image_mask

        # Keep shape (B, 1, ...) when t is per-sample so downstream adaln_modulation
        # projections don't pay for L identical copies.
        t_cond = self.t_embedding(t)
        if t.dim() == 1:
            t_cond = t_cond.unsqueeze(1)
        adaln_input = F.silu(self.adaln_proj(t_cond))

        llm_features = self.llm_cond_norm(llm_features)
        llm_features = self.llm_cond_proj(llm_features) * llm_token_mask

        h = x + llm_features

        image_indicator_embedding = self.embed_image_indicator((indicator == OUTPUT_IMAGE_INDICATOR).to(torch.long))
        h = h + image_indicator_embedding

        cos, sin = self.rotary_emb(position_ids)
        cos = cos.to(h.dtype)
        sin = sin.to(h.dtype)

        # Build the block-diagonal attention mask ONCE here (not 34x, once per block). Under the
        # one-sample-per-row contract (see segment_ids docstring) a batch_size==1 row has no padding, so
        # the mask is all-True; passing None lets SDPA select the flash backend (it rejects a non-null
        # mask) -- numerically a no-op for that case and a real speedup. The guard is the shape-static
        # batch_size==1, NOT segment_ids.unique(): this runs in the EAGER root (no fullgraph boundary
        # here), so the reason to avoid unique() is that .numel() on its data-dependent result forces a
        # per-forward GPU->CPU sync, while batch_size is a free Python int. batch>1 keeps the exact mask.
        # If single-row packing or fixed-length padding is ever added, this batch_size==1 branch must be
        # revisited -- a padded/multi-segment batch-1 row would silently lose its mask here.
        attn_mask = None if batch_size == 1 else (segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)).unsqueeze(1)

        for layer in self.layers:
            h = layer(h, attn_mask=attn_mask, cos=cos, sin=sin, adaln_input=adaln_input)

        out = self.final_layer(h, c=adaln_input)
        return out.to(torch.float32)
