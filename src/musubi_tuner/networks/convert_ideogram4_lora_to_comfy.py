"""Convert an Ideogram 4 LoRA between blissful (sd-scripts style) and ComfyUI formats.

Unlike Z-Image (see convert_z_image_lora_to_comfy.py), no key RENAMES are needed: blissful's
``lora_unet_layers_{N}_attention_{qkv,o}`` / ``lora_unet_layers_{N}_feed_forward_w{1,2,3}`` names
already match ComfyUI's generic LoRA mapping, which registers ``lora_unet_{path}`` for every
``diffusion_model.{path}.weight`` in the model (ComfyUI comfy/lora.py, model_lora_keys_unet), and
ComfyUI's Ideogram4Transformer uses the same module tree (``layers.N.attention.qkv`` etc.). Both
sides keep qkv fused, so there is no split/merge step either.

What DOES need conversion:

1. **Bookkeeping buffers** — blissful saves ``use_dora_flag`` / ``use_rslora_flag`` tensors that
   ComfyUI's loader cannot match (it would log "lora key not loaded" for each). Stripped on
   forward, reconstructed on reverse (DoRA flag from the presence of dora keys; rsLoRA flag False
   because ComfyUI alpha semantics are alpha/r — see 2).
2. **rsLoRA alpha semantics** — blissful applies ``scale = alpha / sqrt(rank)`` when
   ``use_rslora_flag`` is set, but ComfyUI always computes ``alpha / rank``. Forward conversion
   bakes the difference into alpha: ``alpha_comfy = alpha * sqrt(rank)``, making the checkpoint
   self-describing under alpha/r semantics (the flag buffer is then dropped losslessly).
3. **DoRA magnitude key + shape** — blissful saves the magnitude as ``{name}.dora_layer.weight``
   with shape ``(out,)``; ComfyUI expects ``{name}.dora_scale``. The reshape to ``(out, 1)`` is
   load-bearing, not cosmetic: ComfyUI's weight_decompose computes
   ``weight_calc *= dora_scale / weight_norm`` against an ``(out, 1)`` weight_norm, and a 1-D
   ``(out,)`` magnitude broadcasts that division to ``(out, out)`` — a RuntimeError on rectangular
   layers (qkv: 3h x h) and silently wrong math on square ones (attention.o: h x h).

LoHa and LoKr checkpoints (backlog P1-3) are also supported: their adapter keys (``hada_*`` /
``lokr_*``) pass through unchanged — ComfyUI's weight_adapter/{loha,lokr}.py consume them under the
same generic name resolution. The LoKr network-level ``lokr_factor`` buffer is stripped on forward;
``main()`` writes its value into ``ss_lokr_factor`` metadata FIRST (nothing else in production
writes that key — the trainer save path doesn't emit it) and restores the native buffer from that
metadata on reverse. The rsLoRA/DoRA semantics above apply only to the LoRA family (LoKr/LoHa
modules ignore those kwargs at train time).

Usage:
    python src/musubi_tuner/networks/convert_ideogram4_lora_to_comfy.py src.safetensors dst.safetensors
    python src/musubi_tuner/networks/convert_ideogram4_lora_to_comfy.py comfy.safetensors dst.safetensors --reverse
"""

import argparse
import logging
import math

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from musubi_tuner.utils.model_utils import precalculate_safetensors_hashes

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

FLAG_KEYS = ("use_dora_flag", "use_rslora_flag")
# Network-level bookkeeping tensors blissful saves that ComfyUI cannot match ("lora key not loaded"
# noise). lokr_factor is LoKr's factorization persistence — stripping it is safe ONLY because main()
# first writes its value into ss_lokr_factor metadata (which lokr._resolve_factor reads as the
# fallback when the buffer is absent) and restores the buffer from that metadata on reverse. Nothing
# else in production writes ss_lokr_factor, so callers using convert_state_dict() directly must
# handle factor preservation themselves.
BOOKKEEPING_KEYS = FLAG_KEYS + ("lokr_factor",)
BLISSFUL_DORA_SUFFIX = ".dora_layer.weight"
COMFY_DORA_SUFFIX = ".dora_scale"


def convert_state_dict(
    state_dict: dict[str, torch.Tensor], reverse: bool = False
) -> tuple[dict[str, torch.Tensor], dict[str, int]]:
    """Convert a loaded Ideogram 4 LoRA state dict in place of a copy. Returns (new_sd, stats).

    Forward (blissful -> ComfyUI): strip flag buffers, bake rsLoRA scaling into alpha, rename and
    reshape DoRA magnitudes. Reverse (ComfyUI -> blissful): rename/reshape DoRA magnitudes back and
    reconstruct the flag buffers (use_rslora_flag=False — alpha/r semantics are already baked in).
    """
    sd = dict(state_dict)
    stats = {"flags_stripped": 0, "alpha_rescaled": 0, "dora_converted": 0}

    # Checkpoint family. LoHa/LoKr adapter keys pass through unchanged — ComfyUI's
    # weight_adapter/{loha,lokr}.py consume hada_*/lokr_* keys under the same generic
    # lora_unet_{path} name resolution that makes plain LoRA a passthrough. Only the LoRA family
    # carries the rsLoRA/DoRA semantics handled below: LoKrModule absorbs-and-ignores
    # use_rslora/use_dora at train time (lokr.py kwargs), LoHa never receives them.
    has_lora = any(k.endswith(".lora_down.weight") for k in sd)

    if reverse:
        # Operator-error guard: blissful-format markers in a --reverse input mean the file is already
        # blissful format. Proceeding would OVERWRITE its true use_dora_flag with False while leaving
        # .dora_layer.weight keys in place — blissful's loader keys off the flag (lora.py), so the
        # output would load as a plain LoRA with dead DoRA magnitudes. Refuse loudly; legitimate
        # round-trips never hit this because forward output contains no blissful markers.
        blissful_markers = [k for k in sd if k in BOOKKEEPING_KEYS or k.endswith(BLISSFUL_DORA_SUFFIX)]
        if blissful_markers:
            raise ValueError(
                f"--reverse input already contains blissful-format keys (e.g. '{blissful_markers[0]}') — "
                "this is a blissful checkpoint, not a ComfyUI one. Converting it would silently disable "
                "its DoRA magnitudes. Drop --reverse (or pass the correct file)."
            )
        had_dora = False
        for key in list(sd.keys()):
            if key.endswith(COMFY_DORA_SUFFIX):
                had_dora = True
                name = key[: -len(COMFY_DORA_SUFFIX)]
                # clone(): reshape returns a storage-sharing view of the caller's tensor; decouple so
                # no future in-place op on the output can corrupt the input dict.
                sd[f"{name}{BLISSFUL_DORA_SUFFIX}"] = sd.pop(key).reshape(-1).clone()
                stats["dora_converted"] += 1
        if has_lora:
            # Flags are LoRAModule semantics. LoHa networks never carry them; LoKr's loader tolerates
            # their absence — so non-LoRA reverse is a rename-free passthrough here (main() separately
            # restores LoKr's lokr_factor buffer from the ss_lokr_factor metadata it wrote on forward).
            sd["use_dora_flag"] = torch.tensor(had_dora, dtype=torch.bool)
            sd["use_rslora_flag"] = torch.tensor(False, dtype=torch.bool)
        return sd, stats

    use_rslora = bool(sd["use_rslora_flag"].item()) if "use_rslora_flag" in sd else False
    use_dora = bool(sd["use_dora_flag"].item()) if "use_dora_flag" in sd else False

    for key in BOOKKEEPING_KEYS:
        if sd.pop(key, None) is not None:
            stats["flags_stripped"] += 1

    if use_rslora and not has_lora:
        # A LoKr checkpoint can carry use_rslora_flag=True if the user passed the kwarg, but
        # LoKrModule absorbed and IGNORED it at train time — the saved alphas are plain alpha/dim
        # semantics. Rescaling them would corrupt the checkpoint; match training behavior instead.
        logger.info("use_rslora_flag present on a non-LoRA checkpoint — ignored (modules ignored it at train time too)")
        use_rslora = False

    if use_rslora:
        # blissful applies alpha/sqrt(r); ComfyUI will apply alpha/r. alpha *= sqrt(r) makes them equal.
        for key in list(sd.keys()):
            if not key.endswith(".alpha"):
                continue
            name = key[: -len(".alpha")]
            down = sd.get(f"{name}.lora_down.weight")
            if down is None:
                raise ValueError(
                    f"rsLoRA alpha rescale: '{key}' has no matching '{name}.lora_down.weight' to read the rank from. "
                    "Is this a plain-LoRA Ideogram 4 checkpoint?"
                )
            rank = down.shape[0]
            alpha = sd[key]
            sd[key] = torch.tensor(float(alpha.item()) * math.sqrt(rank), dtype=alpha.dtype)
            stats["alpha_rescaled"] += 1
        logger.info(
            f"rsLoRA checkpoint: baked sqrt(rank) into {stats['alpha_rescaled']} alpha value(s) for ComfyUI's alpha/rank semantics"
        )

    for key in list(sd.keys()):
        if key.endswith(BLISSFUL_DORA_SUFFIX):
            name = key[: -len(BLISSFUL_DORA_SUFFIX)]
            # (out,) -> (out, 1): ComfyUI's weight_decompose divides by an (out, 1) weight_norm; a 1-D
            # magnitude broadcasts that to (out, out) — crash on qkv, silently wrong on square layers.
            # clone() decouples the view from the caller's tensor storage.
            sd[f"{name}{COMFY_DORA_SUFFIX}"] = sd.pop(key).reshape(-1, 1).clone()
            stats["dora_converted"] += 1

    if use_dora and has_lora and stats["dora_converted"] == 0:
        raise ValueError("use_dora_flag is set but no '.dora_layer.weight' magnitudes were found — corrupt checkpoint?")
    if use_dora and stats["dora_converted"] > 0:
        logger.warning(
            "DoRA checkpoint: ComfyUI's weight_decompose normalizes by the BASE weight norm on the output axis "
            "(not the merged W+delta norm the trainer uses), so ComfyUI output is a close approximation, not "
            "bit-identical to trainer-side sampling."
        )

    return sd, stats


def main(args):
    logger.info(f"Loading source file {args.src_path}")
    state_dict = {}
    with safe_open(args.src_path, framework="pt") as f:
        metadata = f.metadata()
        for key in f.keys():
            state_dict[key] = f.get_tensor(key)

    # LoKr factor preservation. NOTHING in production writes ss_lokr_factor (the trainer save path
    # never emits it — the _resolve_factor metadata fallback is only fed by callers that pass it),
    # so stripping the lokr_factor buffer WITHOUT writing it here would silently lose any explicit
    # factor (factor=-1 only survives by being the default). The buffer is the authoritative source
    # per _resolve_factor precedence (buffer > metadata), so overwrite — never setdefault.
    if not args.reverse and "lokr_factor" in state_dict:
        metadata = {} if metadata is None else metadata
        metadata["ss_lokr_factor"] = str(int(state_dict["lokr_factor"].item()))
        logger.info(f"LoKr factor {metadata['ss_lokr_factor']} preserved into ss_lokr_factor metadata before buffer strip")

    logger.info("Converting...")
    state_dict, stats = convert_state_dict(state_dict, reverse=args.reverse)

    # Reverse of a ComfyUI-format LoKr file: restore the native lokr_factor buffer from the metadata
    # written on forward, so metadata-blind load paths (and --dim_from_weights reconstruction) see a
    # fully native blissful checkpoint.
    if args.reverse and metadata and "ss_lokr_factor" in metadata and any(".lokr_" in k for k in state_dict):
        state_dict["lokr_factor"] = torch.tensor(int(metadata["ss_lokr_factor"]), dtype=torch.int64)
        logger.info(f"LoKr factor {metadata['ss_lokr_factor']} restored from ss_lokr_factor metadata into the lokr_factor buffer")

    logger.info(f"Flag buffers stripped: {stats['flags_stripped']}")
    logger.info(f"rsLoRA alphas rescaled: {stats['alpha_rescaled']}")
    logger.info(f"DoRA magnitudes converted: {stats['dora_converted']}")

    if metadata is not None:
        logger.info("Calculating hashes and creating metadata...")
        model_hash, legacy_hash = precalculate_safetensors_hashes(state_dict, metadata)
        metadata["sshs_model_hash"] = model_hash
        metadata["sshs_legacy_hash"] = legacy_hash

    logger.info(f"Saving destination file {args.dst_path}")
    save_file(state_dict, args.dst_path, metadata=metadata)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Ideogram 4 LoRA format (blissful/sd-scripts <-> ComfyUI)")
    parser.add_argument("src_path", type=str, help="source path (blissful format; ComfyUI format with --reverse)")
    parser.add_argument("dst_path", type=str, help="destination path")
    parser.add_argument("--reverse", action="store_true", help="convert ComfyUI format back to blissful format")
    main(parser.parse_args())
