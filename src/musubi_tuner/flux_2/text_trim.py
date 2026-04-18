import logging

import torch

# Keep the historical logger name so existing warnings/tests still line up.
logger = logging.getLogger("musubi_tuner.flux_2_train_network")


def compute_trimmed_text_len(full_len: int, ctx_seq_len: torch.Tensor, pad_multiple: int) -> int:
    """Compute the rounded batch-max text length for FLUX.2 cached embeddings."""
    max_real_len = int(ctx_seq_len.max().item())
    if pad_multiple > 1:
        return min(((max_real_len + pad_multiple - 1) // pad_multiple) * pad_multiple, full_len)
    return min(max_real_len, full_len)


def maybe_trim_ctx_vec(
    ctx_vec: torch.Tensor,
    batch: dict[str, torch.Tensor],
    args,
    warned_state: dict,
) -> torch.Tensor:
    """Trim cached text embeddings to the rounded batch-max real sequence length."""
    if getattr(args, "flux2_text_seqlen_mode", "fixed") != "trim":
        return ctx_vec

    ctx_seq_len = batch.get("ctx_seq_len")
    if ctx_seq_len is None:
        if not warned_state.get("fallback_warned", False):
            logger.warning(
                "flux2_text_seqlen_mode='trim' but text cache lacks ctx_seq_len. "
                "Falling back to fixed-512. Re-cache text for trim support."
            )
            warned_state["fallback_warned"] = True
        return ctx_vec

    # Mixed-schema batch: some items have ctx_seq_len, others do not. Trimming to
    # the partial max would silently truncate real tokens from un-annotated items.
    if ctx_seq_len.shape[0] != ctx_vec.shape[0]:
        if not warned_state.get("partial_cache_warned", False):
            logger.warning(
                "flux2_text_seqlen_mode='trim' but batch has mixed cache schemas "
                "(%d/%d items have ctx_seq_len). Skipping trim to avoid truncating "
                "items without seqlen metadata. Re-cache the full dataset to enable trimming.",
                ctx_seq_len.shape[0],
                ctx_vec.shape[0],
            )
            warned_state["partial_cache_warned"] = True
        return ctx_vec

    pad_multiple = getattr(args, "pad_text_seq_len_multiple", None)
    if pad_multiple is None:
        pad_multiple = 32 if getattr(args, "compile", False) else 1
    trimmed_len = compute_trimmed_text_len(ctx_vec.shape[1], ctx_seq_len, pad_multiple)
    if trimmed_len < ctx_vec.shape[1]:
        ctx_vec = ctx_vec[:, :trimmed_len, :]

    return ctx_vec
