# Unified attention function supporting various implementations

import os
from dataclasses import dataclass
import torch
from typing import Optional, Union

from blissful_tuner.blissful_logger import BlissfulLogger

logger = BlissfulLogger(__name__, "green")

try:
    import flash_attn
    from flash_attn.flash_attn_interface import _flash_attn_forward
    from flash_attn.flash_attn_interface import flash_attn_varlen_func
    from flash_attn.flash_attn_interface import flash_attn_func
except ImportError:
    flash_attn = None
    flash_attn_varlen_func = None
    _flash_attn_forward = None
    flash_attn_func = None

try:
    from sageattention import sageattn_varlen, sageattn
except ImportError:
    sageattn_varlen = None
    sageattn = None

try:
    import xformers.ops as xops
except (ImportError, OSError):
    # OSError fires when xformers is installed but its native extension (`mslk.so`,
    # `_C.so`) fails to load — typically because the daily PyTorch rebuild left
    # behind a stale ABI. Treat that the same as "xformers not available" rather
    # than crashing the whole import chain.
    xops = None

try:
    from flash_attn.cute.interface import flash_attn_func as cute_flash_attn_func
    from flash_attn.cute.interface import flash_attn_varlen_func as cute_flash_attn_varlen_func

    CUTE_AVAILABLE = True
except ImportError:
    CUTE_AVAILABLE = False
    cute_flash_attn_func = None
    cute_flash_attn_varlen_func = None


# ---------------------------------------------------------------------------
# Blackwell cuDNN-SDPA guard
#
# cuDNN's fused attention (mha_graph) backward produces NaN / crashes during
# *training* on Blackwell-family GPUs — observed on B300 (sm_103) and reported on
# consumer sm_12x (RTX 5090 / RTX Pro 6000). PyTorch's SDPA dispatcher
# (attn_mode="torch") can auto-select the cuDNN backend, so we disable *only that
# backend* process-wide on Blackwell. FLASH / mem-efficient / math remain — and on
# sm_120 FLASH is both healthy and the fastest path measured, so excluding cuDNN is
# ~free there (it is crash-avoidance, not a speedup, on the data-center cards).
#
# Mechanism note: this flips the process-global torch.backends.cuda.enable_cudnn_sdp
# toggle at import (read by the SDPA dispatcher at torch.compile time — no graph
# break) rather than wrapping each call in an sdpa_kernel() context manager (which
# would sit inside the compiled FLUX.2 region). The global also covers raw SDPA
# call sites outside this module (FLUX.2 VAE attention, HV1.5, Z-Image).
#
# Opt out with BLISSFUL_ALLOW_CUDNN_SDP=1 (e.g. to benchmark, or once a future cuDNN
# fixes the backward).
# ---------------------------------------------------------------------------

# Compute-capability majors for the Blackwell family: 10 = data-center
# (B200 sm_100 / B300 sm_103), 12 = consumer/workstation (sm_120). Hopper (sm_90)
# and earlier are unaffected and keep the cuDNN SDPA backend.
_BLACKWELL_SDP_MAJORS = (10, 12)


def _should_disable_cudnn_sdp(capability_major: Optional[int], *, allow_override: bool) -> bool:
    """Pure decision: should the cuDNN SDPA backend be disabled on this device?

    Returns True only on Blackwell-family GPUs when not overridden. ``capability_major``
    is ``None`` when CUDA is unavailable (→ no guard). Kept side-effect-free so it can
    be unit-tested without a GPU.
    """
    if allow_override:
        return False
    if capability_major is None:
        return False
    return capability_major in _BLACKWELL_SDP_MAJORS


def _apply_blackwell_sdp_guard() -> bool:
    """Disable the cuDNN SDPA backend process-wide on Blackwell. Returns True if applied."""
    allow_override = os.environ.get("BLISSFUL_ALLOW_CUDNN_SDP", "0") == "1"
    try:
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
        else:
            major = minor = None
    except Exception:
        major = minor = None

    if not _should_disable_cudnn_sdp(major, allow_override=allow_override):
        return False
    if not hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        return False

    torch.backends.cuda.enable_cudnn_sdp(False)
    try:
        device_name = torch.cuda.get_device_name()
    except Exception:
        device_name = "unknown"
    logger.info(
        f"Blackwell GPU detected ({device_name}, sm_{major}{minor}) — disabling cuDNN SDPA backend "
        "(buggy attention backward on Blackwell). FLASH / mem-efficient / math remain. "
        "Set BLISSFUL_ALLOW_CUDNN_SDP=1 to re-enable cuDNN."
    )
    return True


_BLACKWELL_SDP_GUARD_APPLIED = _apply_blackwell_sdp_guard()


@torch.compiler.disable
def _cute_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, causal: bool = False):
    """CuTE fixed-length attention (graph-break wrapper for torch.compile)."""
    return cute_flash_attn_func(q, k, v, causal=causal)


# Cache keyed by (device_capability, needs_backward, dtype) for the process lifetime.
_CUTE_PROBE_CACHE: dict[tuple, tuple[bool, str]] = {}


def probe_cute_runtime(
    device: torch.device,
    *,
    needs_backward: bool = False,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[bool, str]:
    """Verify CuTE kernels actually execute on `device`.

    ``CUTE_AVAILABLE`` only confirms the Python side imports. On some GPUs
    (notably SM120 with non-TMA-optimized CuTE builds) kernels JIT-compile on
    first use and can fail there. Call this before model load so training fails
    fast with a clear error instead of mid-step after VAE/TE setup.

    Returns ``(ok, detail)``. ``detail`` is the captured error text when
    ``ok`` is ``False`` and empty when ``ok`` is ``True``. Results are cached.
    """
    if not CUTE_AVAILABLE:
        return False, (
            "CuTE not importable. Install flash-attention with CuTE support "
            "(flash_attn.cute.interface). See docs/cute_attention.md."
        )
    if device.type != "cuda":
        return False, f"CuTE requires a CUDA device, got '{device.type}'."

    cap = torch.cuda.get_device_capability(device)
    key = (cap, needs_backward, dtype)
    if key in _CUTE_PROBE_CACHE:
        return _CUTE_PROBE_CACHE[key]

    # Minimal shape that still triggers the kernel variants most training runs use
    # (head_dim=128, bf16). Kept small so first-run JIT completes in ~seconds.
    B, S, H, D = 1, 128, 4, 128
    try:
        q = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=needs_backward)
        k = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=needs_backward)
        v = torch.randn(B, S, H, D, device=device, dtype=dtype, requires_grad=needs_backward)
        out = _cute_attention(q, k, v, causal=False)
        out_t = out[0] if isinstance(out, tuple) else out
        if needs_backward:
            out_t.float().sum().backward()
        result: tuple[bool, str] = (True, "")
    except Exception as e:
        result = (False, f"{type(e).__name__}: {e}")

    _CUTE_PROBE_CACHE[key] = result
    return result


@torch.compiler.disable
def _cute_attention_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    causal: bool = False,
):
    """CuTE variable-length attention (graph-break wrapper for torch.compile)."""
    return cute_flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_k,
        max_seqlen_q=max_seqlen_q,
        max_seqlen_k=max_seqlen_k,
        causal=causal,
    )


@dataclass
class AttentionParams:
    attn_mode: Optional[str] = None
    split_attn: bool = False
    img_len: Optional[int] = None
    attention_mask: Optional[torch.Tensor] = None
    seqlens: Optional[torch.Tensor] = None
    cu_seqlens: Optional[torch.Tensor] = None
    max_seqlen: Optional[int] = None

    @staticmethod
    def create_attention_params(attn_mode: Optional[str], split_attn: bool) -> "AttentionParams":
        return AttentionParams(attn_mode, split_attn)

    @staticmethod
    def create_attention_params_from_mask(
        attn_mode: Optional[str], split_attn: bool, img_len: Optional[int], attention_mask: Optional[torch.Tensor]
    ) -> "AttentionParams":
        if attention_mask is None:
            # No attention mask provided: assume all tokens are valid
            return AttentionParams(attn_mode, split_attn, None, None, None, None, None)
        else:
            # Note: attention_mask is only for text tokens, not including image tokens
            seqlens = attention_mask.sum(dim=1).to(torch.int32) + img_len  # [B]
            max_seqlen = attention_mask.shape[1] + img_len

            if split_attn:
                # cu_seqlens is not needed for split attention
                return AttentionParams(attn_mode, split_attn, img_len, attention_mask, seqlens, None, max_seqlen)

            # Convert attention mask to cumulative sequence lengths for flash attention
            batch_size = attention_mask.shape[0]
            cu_seqlens = torch.zeros([2 * batch_size + 1], dtype=torch.int32, device=attention_mask.device)
            for i in range(batch_size):
                cu_seqlens[2 * i + 1] = i * max_seqlen + seqlens[i]  # end of valid tokens for query
                cu_seqlens[2 * i + 2] = (i + 1) * max_seqlen  # end of all tokens for query

            # Expand attention mask to include image tokens
            attention_mask = torch.nn.functional.pad(attention_mask, (img_len, 0), value=1)  # [B, img_len + L]

            # attention bias for xformers
            if attn_mode == "xformers":
                seqlens_list = seqlens.cpu().tolist()
                attention_mask = xops.fmha.attn_bias.BlockDiagonalMask.from_seqlens(
                    seqlens_list, seqlens_list, device=attention_mask.device
                )
            elif attn_mode == "torch":
                attention_mask = attention_mask[:, None, None, :].to(torch.bool)  # [B, 1, 1, img_len + L]

            return AttentionParams(attn_mode, split_attn, img_len, attention_mask, seqlens, cu_seqlens, max_seqlen)


def attention(
    qkv_or_q: Union[torch.Tensor, list],
    k: Optional[torch.Tensor] = None,
    v: Optional[torch.Tensor] = None,
    attn_params: Optional[AttentionParams] = None,
    drop_rate: float = 0.0,
    causal: bool = False,
) -> torch.Tensor:
    """
    Compute scaled dot-product attention with variable sequence lengths.

    Handles batches with different sequence lengths by splitting and
    processing each sequence individually.

    Args:
        qkv_or_q: Query tensor [B, L, H, D]. or list of such tensors.
        k: Key tensor [B, L, H, D].
        v: Value tensor [B, L, H, D].
        attn_param: Attention parameters including mask and sequence lengths.
        drop_rate: Attention dropout rate.

    Returns:
        Attention output tensor [B, L, H*D].
    """
    if isinstance(qkv_or_q, list):
        q, k, v = qkv_or_q
        q: torch.Tensor = q
        qkv_or_q.clear()
        qkv_or_q = None
    else:
        q: torch.Tensor = qkv_or_q
        qkv_or_q = None
        assert k is not None and v is not None, "k and v must be provided if qkv_or_q is a tensor"
    if attn_params is None:
        attn_params = AttentionParams.create_attention_params("torch", False)

    # GQA: q may carry more heads than k/v (e.g. Krea 2 = 48 query / 12 kv heads). flash and
    # sageattn group heads natively inside the kernel (verified), and CuTE/FA4 has native GQA
    # (pack_gqa) — so those paths ignore this flag. For the torch (SDPA) and xformers paths we
    # expand k/v to q's head count below instead of passing enable_gqa=True, because enable_gqa
    # forces SDPA onto the slow math kernel (~7x slower than the fused kernels at K2 scale); the
    # repeat is numerically identical. Computed here while q/k/v are still [B, L, H, D], so
    # shape[-2] is the head count (after the transpose below it would be the seq length).
    enable_gqa = q.shape[-2] != k.shape[-2]
    if enable_gqa and (k.shape[-2] != v.shape[-2] or q.shape[-2] % k.shape[-2] != 0):
        # Self-diagnosing guard: GQA expansion repeats k/v by q_heads // kv_heads and assumes k and v
        # share a head count. A future arch violating either would otherwise fail deeper in a kernel
        # with a far less helpful shape error.
        raise ValueError(
            "GQA attention requires equal key/value head counts that divide the query head count, "
            f"got q={q.shape[-2]}, k={k.shape[-2]}, v={v.shape[-2]} heads (q/k/v are [B, L, H, D] here)."
        )

    # If split attn is False, attention mask is provided and all sequence lengths are same, we can trim the sequence
    seqlen_trimmed = False
    # Trim if all seqlens are the same, for attention modes other than flash or sageattn (which can handle masks efficiently)
    if (
        not attn_params.split_attn
        and attn_params.attention_mask is not None
        and attn_params.seqlens is not None
        and (attn_params.attn_mode != "flash" and attn_params.attn_mode != "sageattn")
    ):
        if torch.all(attn_params.seqlens == attn_params.seqlens[0]):
            seqlen = attn_params.seqlens[0].item()
            q = q[:, :seqlen]
            k = k[:, :seqlen]
            v = v[:, :seqlen]
            max_seqlen = attn_params.max_seqlen
            attn_params = AttentionParams.create_attention_params(attn_params.attn_mode, False)  # do not in-place modify
            attn_params.max_seqlen = max_seqlen  # keep max_seqlen for padding
            seqlen_trimmed = True

    # Determine tensor layout based on attention implementation
    if attn_params.attn_mode == "torch" or (
        attn_params.attn_mode == "sageattn" and (attn_params.split_attn or attn_params.cu_seqlens is None)
    ):
        transpose_fn = lambda x: x.transpose(1, 2)  # [B, H, L, D] for SDPA and sageattn with fixed length
        # pad on sequence length dimension
        pad_fn = lambda x, pad_to: torch.nn.functional.pad(x, (0, 0, 0, pad_to - x.shape[-2]), value=0)
    else:
        transpose_fn = lambda x: x  # [B, L, H, D] for other implementations
        # pad on sequence length dimension
        pad_fn = lambda x, pad_to: torch.nn.functional.pad(x, (0, 0, 0, 0, 0, pad_to - x.shape[-3]), value=0)

    # Process each batch element with its valid sequence lengths
    if attn_params.split_attn:
        if attn_params.seqlens is None:
            # If no seqlens provided, assume all tokens are valid
            attn_params = AttentionParams.create_attention_params(attn_params.attn_mode, True)  # do not in-place modify
            attn_params.seqlens = torch.tensor([q.shape[1]] * q.shape[0], device=q.device)
            attn_params.max_seqlen = q.shape[1]
        q = [transpose_fn(q[i : i + 1, : attn_params.seqlens[i]]) for i in range(len(q))]
        k = [transpose_fn(k[i : i + 1, : attn_params.seqlens[i]]) for i in range(len(k))]
        v = [transpose_fn(v[i : i + 1, : attn_params.seqlens[i]]) for i in range(len(v))]
    else:
        q = transpose_fn(q)
        k = transpose_fn(k)
        v = transpose_fn(v)

    if attn_params.attn_mode == "torch":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                qi, ki, vi = q[i], k[i], v[i]
                if enable_gqa:  # expand k/v heads to avoid SDPA's slow enable_gqa math path
                    g = qi.shape[1] // ki.shape[1]  # [B, H, L, D] -> heads at dim 1
                    ki = ki.repeat_interleave(g, dim=1)
                    vi = vi.repeat_interleave(g, dim=1)
                x_i = torch.nn.functional.scaled_dot_product_attention(qi, ki, vi, dropout_p=drop_rate, is_causal=causal)
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, H, L, D
            x = torch.cat(x, dim=0)
            q, k, v = None, None, None

        else:
            if causal and attn_params.attention_mask is not None:
                raise ValueError("Causal attention is not supported with an explicit attention_mask in unified attention.")
            if enable_gqa:  # expand k/v heads to avoid SDPA's slow enable_gqa math path
                g = q.shape[1] // k.shape[1]  # [B, H, L, D] -> heads at dim 1
                k = k.repeat_interleave(g, dim=1)
                v = v.repeat_interleave(g, dim=1)
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_params.attention_mask, dropout_p=drop_rate, is_causal=causal
            )
            q, k, v = None, None, None

    elif attn_params.attn_mode == "xformers":
        if causal:
            raise ValueError("Causal attention is not currently supported for xformers in unified attention.")
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                qi, ki, vi = q[i], k[i], v[i]
                if enable_gqa:  # expand k/v heads to q's count; xformers 4D has no native GQA ([B, L, H, D] -> dim 2)
                    g = qi.shape[2] // ki.shape[2]
                    ki = ki.repeat_interleave(g, dim=2)
                    vi = vi.repeat_interleave(g, dim=2)
                x_i = xops.memory_efficient_attention(qi, ki, vi, p=drop_rate)
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, L, H, D
            x = torch.cat(x, dim=0)
            q, k, v = None, None, None

        else:
            if enable_gqa:  # expand k/v heads to q's count; xformers 4D has no native GQA ([B, L, H, D] -> dim 2)
                g = q.shape[2] // k.shape[2]
                k = k.repeat_interleave(g, dim=2)
                v = v.repeat_interleave(g, dim=2)
            x = xops.memory_efficient_attention(q, k, v, attn_bias=attn_params.attention_mask, p=drop_rate)
            q, k, v = None, None, None

    elif attn_params.attn_mode == "sageattn":
        if causal:
            raise ValueError("Causal attention is not currently supported for sageattention in unified attention.")
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                # HND seems to cause an error
                x_i = sageattn(q[i], k[i], v[i])  # B, H, L, D. No dropout support
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, H, L, D
            x = torch.cat(x, dim=0)
            q, k, v = None, None, None
        elif attn_params.cu_seqlens is None:  # all tokens are valid
            x = sageattn(q, k, v)  # B, L, H, D. No dropout support
            q, k, v = None, None, None
        else:
            # Reshape to [(bxs), a, d]
            batch_size, seqlen = q.shape[0], q.shape[1]
            # reshape (not view): callers may pass non-contiguous q/k/v (e.g. Krea 2 transposes
            # [B,H,L,D]->[B,L,H,D] after RoPE); reshape copies only when needed, view-equivalent otherwise.
            q = q.reshape(q.shape[0] * q.shape[1], *q.shape[2:])  # [B*L, H, D]
            k = k.reshape(k.shape[0] * k.shape[1], *k.shape[2:])  # [B*L, H, D]
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])  # [B*L, H, D]

            # Assume cu_seqlens_q == cu_seqlens_kv and max_seqlen_q == max_seqlen_kv. No dropout support
            x = sageattn_varlen(
                q, k, v, attn_params.cu_seqlens, attn_params.cu_seqlens, attn_params.max_seqlen, attn_params.max_seqlen
            )
            q, k, v = None, None, None

            # Reshape x with shape [(bxs), a, d] to [b, s, a, d]
            x = x.view(batch_size, seqlen, x.shape[-2], x.shape[-1])  # B, L, H, D

    elif attn_params.attn_mode == "flash":
        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                # HND seems to cause an error
                if causal:
                    x_i = flash_attn_func(q[i], k[i], v[i], drop_rate, causal=True)  # B, L, H, D
                else:
                    x_i = flash_attn_func(q[i], k[i], v[i], drop_rate)  # B, L, H, D
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(x_i, attn_params.max_seqlen))  # B, L, H, D
            x = torch.cat(x, dim=0)
            q, k, v = None, None, None
        elif attn_params.cu_seqlens is None:  # all tokens are valid
            if causal:
                x = flash_attn_func(q, k, v, drop_rate, causal=True)  # B, L, H, D
            else:
                x = flash_attn_func(q, k, v, drop_rate)  # B, L, H, D
            q, k, v = None, None, None
        else:
            # Reshape to [(bxs), a, d]
            batch_size, seqlen = q.shape[0], q.shape[1]
            # reshape (not view): callers may pass non-contiguous q/k/v (e.g. Krea 2 transposes
            # [B,H,L,D]->[B,L,H,D] after RoPE); reshape copies only when needed, view-equivalent otherwise.
            q = q.reshape(q.shape[0] * q.shape[1], *q.shape[2:])  # [B*L, H, D]
            k = k.reshape(k.shape[0] * k.shape[1], *k.shape[2:])  # [B*L, H, D]
            v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])  # [B*L, H, D]

            # Assume cu_seqlens_q == cu_seqlens_kv and max_seqlen_q == max_seqlen_kv
            if causal:
                x = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    attn_params.cu_seqlens,
                    attn_params.cu_seqlens,
                    attn_params.max_seqlen,
                    attn_params.max_seqlen,
                    drop_rate,
                    causal=True,
                )
            else:
                x = flash_attn_varlen_func(
                    q,
                    k,
                    v,
                    attn_params.cu_seqlens,
                    attn_params.cu_seqlens,
                    attn_params.max_seqlen,
                    attn_params.max_seqlen,
                    drop_rate,
                )
            q, k, v = None, None, None

            # Reshape x with shape [(bxs), a, d] to [b, s, a, d]
            x = x.view(batch_size, seqlen, x.shape[-2], x.shape[-1])  # B, L, H, D

    elif attn_params.attn_mode == "cute":
        if not CUTE_AVAILABLE:
            raise ImportError(
                "CuTE not available. Requires flash-attention built with CuTE support (flash_attn.cute.interface). "
                "See docs/cute_attention.md for requirements."
            )
        if drop_rate != 0.0:
            raise ValueError("CuTE attention does not support dropout. Set attention dropout to 0.0.")

        if attn_params.split_attn:
            x = []
            for i in range(len(q)):
                out, _ = _cute_attention(q[i], k[i], v[i], causal=causal)  # B=1, L, H, D
                q[i] = None
                k[i] = None
                v[i] = None
                x.append(pad_fn(out, attn_params.max_seqlen))  # B, L, H, D
            x = torch.cat(x, dim=0)
            q, k, v = None, None, None
        elif attn_params.cu_seqlens is None:  # all tokens are valid
            out, _ = _cute_attention(q, k, v, causal=causal)  # B, L, H, D
            x = out
            q, k, v = None, None, None
        else:
            # If there is no padding, bypass the varlen path entirely (and avoid SM90 varlen-backward limits).
            no_padding = (
                attn_params.seqlens is not None
                and attn_params.max_seqlen is not None
                and torch.all(attn_params.seqlens == attn_params.max_seqlen)
            )
            if no_padding:
                out, _ = _cute_attention(q, k, v, causal=causal)  # B, L, H, D
                x = out
                q, k, v = None, None, None
            else:
                # Hopper (SM90) does not currently support CuTE varlen backward.
                requires_backward = torch.is_grad_enabled() and (q.requires_grad or k.requires_grad or v.requires_grad)
                if requires_backward and q.is_cuda:
                    major, _minor = torch.cuda.get_device_capability(q.device)
                    if major == 9:
                        raise ValueError(
                            "CuTE varlen backward is not supported on Hopper (SM90). "
                            "Use --split_attn (forces fixed-length CuTE per-sample) or switch to --flash_attn/--sage_attn."
                        )

                # Reshape to [(bxs), a, d]
                batch_size, seqlen = q.shape[0], q.shape[1]
                # reshape (not view): callers may pass non-contiguous q/k/v (e.g. Krea 2 transposes
                # [B,H,L,D]->[B,L,H,D] after RoPE); reshape copies only when needed, view-equivalent otherwise.
                q = q.reshape(q.shape[0] * q.shape[1], *q.shape[2:])  # [B*L, H, D]
                k = k.reshape(k.shape[0] * k.shape[1], *k.shape[2:])  # [B*L, H, D]
                v = v.reshape(v.shape[0] * v.shape[1], *v.shape[2:])  # [B*L, H, D]

                # Assume cu_seqlens_q == cu_seqlens_kv and max_seqlen_q == max_seqlen_kv.
                out, _ = _cute_attention_varlen(
                    q,
                    k,
                    v,
                    cu_seqlens_q=attn_params.cu_seqlens,
                    cu_seqlens_k=attn_params.cu_seqlens,
                    max_seqlen_q=attn_params.max_seqlen,
                    max_seqlen_k=attn_params.max_seqlen,
                    causal=causal,
                )
                q, k, v = None, None, None

                # Reshape out with shape [(bxs), a, d] to [b, s, a, d]
                x = out.view(batch_size, seqlen, out.shape[-2], out.shape[-1])  # B, L, H, D

    else:
        # Currently only PyTorch SDPA and xformers are implemented
        raise ValueError(f"Unsupported attention mode: {attn_params.attn_mode}")

    x = transpose_fn(x)  # [B, L, H, D]
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, L, H*D]

    if seqlen_trimmed:
        x = torch.nn.functional.pad(x, (0, 0, 0, attn_params.max_seqlen - x.shape[1]), value=0)  # pad back to max_seqlen

    return x
