import argparse
import hashlib
from io import BytesIO
from typing import Any, Callable, Optional
import logging
import safetensors.torch
import torch

logger = logging.getLogger(__name__)


def model_hash(filename):
    """Old model hash used by stable-diffusion-webui"""
    try:
        with open(filename, "rb") as file:
            m = hashlib.sha256()

            file.seek(0x100000)
            m.update(file.read(0x10000))
            return m.hexdigest()[0:8]
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:  # Linux?
        return "IsADirectory"
    except PermissionError:  # Windows
        return "IsADirectory"


def calculate_sha256(filename):
    """New model hash used by stable-diffusion-webui"""
    try:
        hash_sha256 = hashlib.sha256()
        blksize = 1024 * 1024

        with open(filename, "rb") as f:
            for chunk in iter(lambda: f.read(blksize), b""):
                hash_sha256.update(chunk)

        return hash_sha256.hexdigest()
    except FileNotFoundError:
        return "NOFILE"
    except IsADirectoryError:  # Linux?
        return "IsADirectory"
    except PermissionError:  # Windows
        return "IsADirectory"


def addnet_hash_legacy(b):
    """Old model hash used by sd-webui-additional-networks for .safetensors format files"""
    m = hashlib.sha256()

    b.seek(0x100000)
    m.update(b.read(0x10000))
    return m.hexdigest()[0:8]


def addnet_hash_safetensors(b):
    """New model hash used by sd-webui-additional-networks for .safetensors format files"""
    hash_sha256 = hashlib.sha256()
    blksize = 1024 * 1024

    b.seek(0)
    header = b.read(8)
    n = int.from_bytes(header, "little")

    offset = n + 8
    b.seek(offset)
    for chunk in iter(lambda: b.read(blksize), b""):
        hash_sha256.update(chunk)

    return hash_sha256.hexdigest()


def precalculate_safetensors_hashes(tensors, metadata):
    """Precalculate the model hashes needed by sd-webui-additional-networks to
    save time on indexing the model later."""

    # Because writing user metadata to the file can change the result of
    # sd_models.model_hash(), only retain the training metadata for purposes of
    # calculating the hash, as they are meant to be immutable
    metadata = {k: v for k, v in metadata.items() if k.startswith("ss_")}

    bytes = safetensors.torch.save(tensors, metadata)
    b = BytesIO(bytes)

    model_hash = addnet_hash_safetensors(b)
    legacy_hash = addnet_hash_legacy(b)
    return model_hash, legacy_hash


def dtype_to_str(dtype: torch.dtype) -> str:
    # get name of the dtype
    dtype_name = str(dtype).split(".")[-1]
    return dtype_name


# Cache keys are formatted "{stem}_{dtype_to_str(dtype)}". Several dtype names contain underscores
# (float8_e4m3fn, float8_e5m2, the *uz variants), so a naive rsplit("_", 1) mangles them into "...float8".
# Match known dtype suffixes LONGEST-FIRST instead. Built from torch dtypes, getattr-guarded for older builds.
_CACHE_DTYPE_STRINGS = sorted(
    {
        dtype_to_str(d)
        for d in (
            getattr(torch, name, None)
            for name in (
                "float64",
                "float32",
                "float16",
                "bfloat16",
                "float8_e4m3fn",
                "float8_e5m2",
                "float8_e4m3fnuz",
                "float8_e5m2fnuz",
                "int8",
                "int16",
                "int32",
                "int64",
                "uint8",
                "bool",
                "complex32",
                "complex64",
                "complex128",
            )
        )
        if d is not None
    },
    key=len,
    reverse=True,
)


def strip_dtype_suffix(key: str) -> str:
    """Strip a trailing ``_<dtype>`` suffix from a cache key, matching known dtype strings LONGEST-FIRST.

    Robust to multi-underscore dtype names (e.g. ``float8_e4m3fn``) that a naive ``rsplit("_", 1)`` would
    mangle into ``"...float8"``. Returns the key unchanged if it carries no recognized dtype suffix.
    """
    for dtype_str in _CACHE_DTYPE_STRINGS:
        suffix = "_" + dtype_str
        if key.endswith(suffix):
            return key[: -len(suffix)]
    return key


def str_to_dtype(s: Optional[str], default_dtype: Optional[torch.dtype] = None) -> torch.dtype:
    """
    Convert a string to a torch.dtype

    Args:
        s: string representation of the dtype
        default_dtype: default dtype to return if s is None

    Returns:
        torch.dtype: the corresponding torch.dtype

    Raises:
        ValueError: if the dtype is not supported

    Examples:
        >>> str_to_dtype("float32")
        torch.float32
        >>> str_to_dtype("fp32")
        torch.float32
        >>> str_to_dtype("float16")
        torch.float16
        >>> str_to_dtype("fp16")
        torch.float16
        >>> str_to_dtype("bfloat16")
        torch.bfloat16
        >>> str_to_dtype("bf16")
        torch.bfloat16
        >>> str_to_dtype("fp8")
        torch.float8_e4m3fn
        >>> str_to_dtype("fp8_e4m3fn")
        torch.float8_e4m3fn
        >>> str_to_dtype("fp8_e4m3fnuz")
        torch.float8_e4m3fnuz
        >>> str_to_dtype("fp8_e5m2")
        torch.float8_e5m2
        >>> str_to_dtype("fp8_e5m2fnuz")
        torch.float8_e5m2fnuz
    """
    if s is None:
        return default_dtype
    if s in ["bf16", "bfloat16"]:
        return torch.bfloat16
    elif s in ["fp16", "float16"]:
        return torch.float16
    elif s in ["fp32", "float32", "float"]:
        return torch.float32
    elif s in ["fp8_e4m3fn", "e4m3fn", "float8_e4m3fn"]:
        return torch.float8_e4m3fn
    elif s in ["fp8_e4m3fnuz", "e4m3fnuz", "float8_e4m3fnuz"]:
        return torch.float8_e4m3fnuz
    elif s in ["fp8_e5m2", "e5m2", "float8_e5m2"]:
        return torch.float8_e5m2
    elif s in ["fp8_e5m2fnuz", "e5m2fnuz", "float8_e5m2fnuz"]:
        return torch.float8_e5m2fnuz
    elif s in ["fp8", "float8"]:
        return torch.float8_e4m3fn  # default fp8
    else:
        raise ValueError(f"Unsupported dtype: {s}")


def to_device(x: Any, device: torch.device) -> Any:
    if isinstance(x, torch.Tensor):
        return x.to(device)
    elif isinstance(x, list):
        return [to_device(elem, device) for elem in x]
    elif isinstance(x, tuple):
        return tuple(to_device(elem, device) for elem in x)
    elif isinstance(x, dict):
        return {k: to_device(v, device) for k, v in x.items()}
    else:
        return x


def to_cpu(x: Any) -> Any:
    """
    Recursively moves torch.Tensor objects (and containers thereof) to CPU.

    Args:
        x: A torch.Tensor, or a (possibly nested) list, tuple, or dict containing tensors.

    Returns:
        The same structure as x, with all torch.Tensor objects moved to CPU.
        Non-tensor objects are returned unchanged.
    """
    if isinstance(x, torch.Tensor):
        return x.cpu()
    elif isinstance(x, list):
        return [to_cpu(elem) for elem in x]
    elif isinstance(x, tuple):
        return tuple(to_cpu(elem) for elem in x)
    elif isinstance(x, dict):
        return {k: to_cpu(v) for k, v in x.items()}
    else:
        return x


def create_cpu_offloading_wrapper(func: Callable, device: torch.device) -> Callable:
    """
    Create a wrapper function that offloads inputs to CPU before calling the original function
    and moves outputs back to the specified device.

    Args:
        func: The original function to wrap.
        device: The device to move outputs back to.

    Returns:
        A wrapped function that offloads inputs to CPU and moves outputs back to the specified device.
    """

    def wrapper(orig_func: Callable) -> Callable:
        def custom_forward(*inputs):
            nonlocal device, orig_func
            cuda_inputs = to_device(inputs, device)
            outputs = orig_func(*cuda_inputs)
            return to_cpu(outputs)

        return custom_forward

    return wrapper(func)


def disable_linear_from_compile(module: torch.nn.Module):
    """Monkey-patch to disable torch.compile for all Linear layers (if the class name ends with 'Linear') in the given module."""
    for sub_module in module.modules():
        # if isinstance(sub_module, torch.nn.Linear):
        if sub_module.__class__.__name__.endswith("Linear"):
            if not hasattr(sub_module, "_forward_before_disable_compile"):
                sub_module._forward_before_disable_compile = sub_module.forward
                sub_module._eager_forward = torch._dynamo.disable()(sub_module.forward)
            sub_module.forward = sub_module._eager_forward  # override forward to disable compile


def compile_transformer(
    args: argparse.Namespace,
    transformer: torch.nn.Module,
    target_blocks: list[torch.nn.ModuleList | list[torch.nn.Module]],
    disable_linear: bool,
) -> torch.nn.Module:
    if disable_linear:
        logger.info("Disable linear from torch.compile for swap blocks...")
        for blocks in target_blocks:
            for block in blocks:
                disable_linear_from_compile(block)

    compile_dynamic = None
    if args.compile_dynamic is not None:
        compile_dynamic = {"true": True, "false": False, "auto": None}[args.compile_dynamic.lower()]

    logger.info(
        f"Compiling DiT model with torch.compile: backend={args.compile_backend}, mode={args.compile_mode}, dynamic={compile_dynamic}, fullgraph={args.compile_fullgraph}"
    )

    if args.compile_cache_size_limit is not None:
        torch._dynamo.config.cache_size_limit = args.compile_cache_size_limit

    for blocks in target_blocks:
        for i, block in enumerate(blocks):
            block = torch.compile(
                block,
                backend=args.compile_backend,
                mode=args.compile_mode,
                dynamic=compile_dynamic,
                fullgraph=args.compile_fullgraph,
            )
            blocks[i] = block
    return transformer


def count_dynamo_cache_entries(model: torch.nn.Module) -> int:
    """Sum of Dynamo cache entries across all unique compiled forwards in `model`.

    Walks `model.modules()` once, finds each `OptimizedModule` (identified by a `_orig_mod`
    attribute pointing to a *different* module), and sums `_get_total_cache_entry_count`
    over the **distinct code objects** of their forwards. Dedup matters: N instances of the
    same block class share one Dynamo cache keyed on `Class.forward.__code__`, so summing
    per-instance would over-report by Nx. Returns 0 when the private API is unavailable
    (pre-2.13 builds) or no compiled blocks are present.
    """
    try:
        from torch._dynamo.eval_frame import _get_total_cache_entry_count
    except (ImportError, AttributeError):
        return 0

    seen_codes: set = set()
    total = 0
    for module in model.modules():
        orig = getattr(module, "_orig_mod", None)
        # Skip non-compiled modules and the accelerator self-reference set on the
        # top-level transformer (hv_train_network.py: `transformer.__dict__["_orig_mod"] = transformer`).
        if orig is None or orig is module:
            continue
        try:
            fn = orig.forward
            code = getattr(fn, "__code__", None)
            if code is None or code in seen_codes:
                continue
            seen_codes.add(code)
            total += _get_total_cache_entry_count(fn)
        except Exception:
            # Private API may shift across daily builds; one bad block must not kill telemetry.
            pass
    return total
