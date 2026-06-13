import argparse
import json
from blissful_tuner.blissful_logger import BlissfulLogger
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Callable

import accelerate
import torch

from musubi_tuner.utils import huggingface_utils
from musubi_tuner.utils.model_utils import str_to_dtype

logger = BlissfulLogger(__name__, "green")

SENSITIVE_MANIFEST_ARGS = {"wandb_api_key", "huggingface_token"}
RUN_MANIFEST_ENV_KEYS = (
    "PYTHONUNBUFFERED",
    "HF_HOME",
    "TRANSFORMERS_CACHE",
    "TORCHINDUCTOR_CACHE_DIR",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "CUDA_MODULE_LOADING",
    "TOKENIZERS_PARALLELISM",
    "CUDA_VISIBLE_DEVICES",
    "ACCELERATE_CONFIG_FILE",
    "CACHE_MASK_GAMMA",
    "CACHE_MASK_MIN_WEIGHT",
    "RUN_MANIFEST_LAUNCHER_SCRIPT",
    "RUN_MANIFEST_ENV_SCRIPT",
    "RUN_MANIFEST_INVOCATION_NOTE",
)


# checkpointファイル名
EPOCH_STATE_NAME = "{}-{:06d}-state"
EPOCH_FILE_NAME = "{}-{:06d}"
EPOCH_DIFFUSERS_DIR_NAME = "{}-{:06d}"
LAST_STATE_NAME = "{}-state"
STEP_STATE_NAME = "{}-step{:08d}-state"
STEP_FILE_NAME = "{}-step{:08d}"
STEP_DIFFUSERS_DIR_NAME = "{}-step{:08d}"


def get_sanitized_config_or_none(args: argparse.Namespace):
    # if `--log_config` is enabled, return args for logging. if not, return None.
    # when `--log_config is enabled, filter out sensitive values from args
    # if wandb is not enabled, the log is not exposed to the public, but it is fine to filter out sensitive values to be safe

    if not args.log_config:
        return None

    sensitive_args = ["wandb_api_key", "huggingface_token"]
    sensitive_path_args = [
        "dit",
        "vae",
        "text_encoder1",
        "text_encoder2",
        "image_encoder",
        "base_weights",
        "network_weights",
        "output_dir",
        "logging_dir",
    ]
    filtered_args = {}
    for k, v in vars(args).items():
        # filter out sensitive values and convert to string if necessary
        if k not in sensitive_args + sensitive_path_args:
            # Accelerate values need to have type `bool`,`str`, `float`, `int`, or `None`.
            if v is None or isinstance(v, bool) or isinstance(v, str) or isinstance(v, float) or isinstance(v, int):
                filtered_args[k] = v
            # accelerate does not support lists
            elif isinstance(v, list):
                filtered_args[k] = f"{v}"
            # accelerate does not support objects
            elif isinstance(v, object):
                filtered_args[k] = f"{v}"

    return filtered_args


def _serialize_manifest_value(value: Any) -> Any:
    if value is None or isinstance(value, bool | int | float | str):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (list, tuple, set)):
        return [_serialize_manifest_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _serialize_manifest_value(v) for k, v in value.items()}
    return str(value)


def _resolve_optional_path(path_like: str | os.PathLike[str] | None, *, suffix: str | None = None) -> Path | None:
    if not path_like:
        return None

    path = Path(path_like).expanduser()
    candidates = [path]
    if suffix and path.suffix != suffix:
        candidates.append(path.with_suffix(suffix))

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    return None


def _copy_snapshot_if_present(src: Path | None, dest_dir: Path, dest_name: str) -> dict[str, str] | None:
    if src is None or not src.exists() or not src.is_file():
        return None

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest_path = dest_dir / dest_name
    dest_path.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")

    return {
        "source_path": str(src),
        "snapshot_path": str(dest_path),
    }


def get_run_manifest(args: argparse.Namespace, *, tracker_name: str | None = None) -> dict[str, Any]:
    manifest_args: dict[str, Any] = {}
    for key, value in vars(args).items():
        if key in SENSITIVE_MANIFEST_ARGS:
            continue
        manifest_args[key] = _serialize_manifest_value(value)

    env = {key: os.environ[key] for key in RUN_MANIFEST_ENV_KEYS if key in os.environ}

    return {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
        "cwd": os.getcwd(),
        "python_executable": sys.executable,
        "argv": sys.argv,
        "tracker_name": tracker_name,
        "args": manifest_args,
        "env": env,
    }


def write_run_manifest(
    args: argparse.Namespace,
    *,
    run_dir: str | os.PathLike[str] | None,
    tracker_name: str | None = None,
) -> dict[str, str]:
    output_dir = Path(args.output_dir).expanduser().resolve() if getattr(args, "output_dir", None) else None
    primary_run_dir = Path(run_dir).expanduser().resolve() if run_dir else None
    run_id = primary_run_dir.name if primary_run_dir else time.strftime("%Y%m%d%H%M%S", time.localtime())

    roots: list[Path] = []
    if primary_run_dir is not None:
        roots.append(primary_run_dir)
    if output_dir is not None:
        roots.append(output_dir / "run_manifests" / run_id)

    unique_roots: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        root_key = str(root)
        if root_key in seen:
            continue
        seen.add(root_key)
        unique_roots.append(root)

    if not unique_roots:
        return {}

    manifest = get_run_manifest(args, tracker_name=tracker_name)
    training_config = _resolve_optional_path(getattr(args, "config_file", None), suffix=".toml")
    dataset_config = _resolve_optional_path(getattr(args, "dataset_config", None))
    sample_prompts = _resolve_optional_path(getattr(args, "sample_prompts", None))
    tracker_config = _resolve_optional_path(getattr(args, "log_tracker_config", None))
    launcher_script = _resolve_optional_path(os.environ.get("RUN_MANIFEST_LAUNCHER_SCRIPT"))
    launcher_env = _resolve_optional_path(os.environ.get("RUN_MANIFEST_ENV_SCRIPT"))
    invocation_note = _resolve_optional_path(os.environ.get("RUN_MANIFEST_INVOCATION_NOTE"))

    created_paths: dict[str, str] = {}
    for root in unique_roots:
        root.mkdir(parents=True, exist_ok=True)
        snapshot_dir = root / "run_snapshot"
        manifest_path = root / "run_manifest.json"

        snapshots = {
            "training_config": _copy_snapshot_if_present(training_config, snapshot_dir, "training_config.toml"),
            "dataset_config": _copy_snapshot_if_present(dataset_config, snapshot_dir, "dataset_config.toml"),
            "sample_prompts": _copy_snapshot_if_present(sample_prompts, snapshot_dir, "sample_prompts.txt"),
            "tracker_config": _copy_snapshot_if_present(tracker_config, snapshot_dir, "tracker_config.toml"),
            "launcher_script": _copy_snapshot_if_present(launcher_script, snapshot_dir, "launcher_script.sh"),
            "launcher_env": _copy_snapshot_if_present(launcher_env, snapshot_dir, "launcher_env.sh"),
            "invocation_note": _copy_snapshot_if_present(invocation_note, snapshot_dir, "invocation_note.txt"),
        }

        manifest_to_write = dict(manifest)
        manifest_to_write["snapshots"] = {k: v for k, v in snapshots.items() if v is not None}
        manifest_path.write_text(json.dumps(manifest_to_write, indent=2, sort_keys=True), encoding="utf-8")

        if root == unique_roots[0]:
            created_paths["manifest_json"] = str(manifest_path)
            created_paths["snapshot_dir"] = str(snapshot_dir)

    return created_paths


class LossRecorder:
    def __init__(self):
        self.loss_list: list[float] = []
        self.loss_total: float = 0.0

    def add(self, *, epoch: int, step: int, loss: float) -> None:
        if epoch == 0:
            self.loss_list.append(loss)
        else:
            while len(self.loss_list) <= step:
                self.loss_list.append(0.0)
            self.loss_total -= self.loss_list[step]
            self.loss_list[step] = loss
        self.loss_total += loss

    @property
    def moving_average(self) -> float:
        return self.loss_total / len(self.loss_list)


def get_epoch_ckpt_name(model_name, epoch_no: int):
    return EPOCH_FILE_NAME.format(model_name, epoch_no) + ".safetensors"


def get_step_ckpt_name(model_name, step_no: int):
    return STEP_FILE_NAME.format(model_name, step_no) + ".safetensors"


def get_last_ckpt_name(model_name):
    return model_name + ".safetensors"


def get_remove_epoch_no(args: argparse.Namespace, epoch_no: int):
    if args.save_last_n_epochs is None:
        return None

    remove_epoch_no = epoch_no - args.save_every_n_epochs * args.save_last_n_epochs
    if remove_epoch_no < 0:
        return None
    return remove_epoch_no


def get_remove_step_no(args: argparse.Namespace, step_no: int):
    if args.save_last_n_steps is None:
        return None

    # calculate the step number to remove from the last_n_steps and save_every_n_steps
    # e.g. if save_every_n_steps=10, save_last_n_steps=30, at step 50, keep 30 steps and remove step 10
    remove_step_no = step_no - args.save_last_n_steps - 1
    remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)
    if remove_step_no < 0:
        return None
    return remove_step_no


def save_and_remove_state_on_epoch_end(args: argparse.Namespace, accelerator: accelerate.Accelerator, epoch_no: int):
    model_name = args.output_name

    logger.info("")
    logger.info(f"saving state at epoch {epoch_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, epoch_no))
    accelerator.save_state(state_dir)
    if args.save_state_to_huggingface:
        logger.info("uploading state to huggingface.")
        huggingface_utils.upload(args, state_dir, "/" + EPOCH_STATE_NAME.format(model_name, epoch_no))

    last_n_epochs = args.save_last_n_epochs_state if args.save_last_n_epochs_state else args.save_last_n_epochs
    if last_n_epochs is not None:
        remove_epoch_no = epoch_no - args.save_every_n_epochs * last_n_epochs
        state_dir_old = os.path.join(args.output_dir, EPOCH_STATE_NAME.format(model_name, remove_epoch_no))
        if os.path.exists(state_dir_old):
            logger.info(f"removing old state: {state_dir_old}")
            shutil.rmtree(state_dir_old)


def save_and_remove_state_stepwise(args: argparse.Namespace, accelerator: accelerate.Accelerator, step_no: int):
    model_name = args.output_name

    logger.info("")
    logger.info(f"saving state at step {step_no}")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, STEP_STATE_NAME.format(model_name, step_no))
    accelerator.save_state(state_dir)
    if args.save_state_to_huggingface:
        logger.info("uploading state to huggingface.")
        huggingface_utils.upload(args, state_dir, "/" + STEP_STATE_NAME.format(model_name, step_no))

    last_n_steps = args.save_last_n_steps_state if args.save_last_n_steps_state else args.save_last_n_steps
    if last_n_steps is not None:
        # last_n_steps前のstep_noから、save_every_n_stepsの倍数のstep_noを計算して削除する
        remove_step_no = step_no - last_n_steps - 1
        remove_step_no = remove_step_no - (remove_step_no % args.save_every_n_steps)

        if remove_step_no > 0:
            state_dir_old = os.path.join(args.output_dir, STEP_STATE_NAME.format(model_name, remove_step_no))
            if os.path.exists(state_dir_old):
                logger.info(f"removing old state: {state_dir_old}")
                shutil.rmtree(state_dir_old)


def save_state_on_train_end(args: argparse.Namespace, accelerator: accelerate.Accelerator):
    model_name = args.output_name

    logger.info("")
    logger.info("saving last state.")
    os.makedirs(args.output_dir, exist_ok=True)

    state_dir = os.path.join(args.output_dir, LAST_STATE_NAME.format(model_name))
    accelerator.save_state(state_dir)

    if args.save_state_to_huggingface:
        logger.info("uploading last state to huggingface.")
        huggingface_utils.upload(args, state_dir, "/" + LAST_STATE_NAME.format(model_name))


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def resolve_save_dtype(save_precision: str | None, full_fp16: bool = False, full_bf16: bool = False) -> torch.dtype:
    """Resolve the dtype for saving network weights.

    Explicit --save_precision wins; otherwise follow full_fp16/full_bf16 so the
    saved weights match the training precision; otherwise fp32, the precision
    the network weights are actually trained in.
    """
    if save_precision is not None:
        return str_to_dtype(save_precision)
    if full_fp16:
        return torch.float16
    if full_bf16:
        return torch.bfloat16
    return torch.float32
