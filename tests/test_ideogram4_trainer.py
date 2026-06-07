from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from musubi_tuner.ideogram4 import constants as i4c
from musubi_tuner.ideogram4_train_network import (
    Ideogram4NetworkTrainer,
    ideogram4_setup_parser,
    neutralize_unused_fp8_args,
)


class _FakeDataset:
    def __init__(self, cache_directory, files):
        self.cache_directory = cache_directory
        self._files = files

    def get_all_latent_cache_files(self):
        return self._files


class _FakeGroup:
    def __init__(self, datasets):
        self.datasets = datasets


def _write_stale_cache(path) -> str:
    # A raw / fork / stale cache the shared reader would happily load (no latent_norm flag).
    save_file(
        {"latents_4x4_bfloat16": torch.zeros(128, 4, 4, dtype=torch.bfloat16)},
        str(path),
        metadata={"architecture": "ideogram4"},
    )
    return str(path)


def _write_valid_cache(path) -> str:
    save_file(
        {"latents_4x4_bfloat16": torch.zeros(128, 4, 4, dtype=torch.bfloat16)},
        str(path),
        metadata={
            i4c.IDEOGRAM4_LATENT_NORM_METADATA_KEY: "true",
            i4c.IDEOGRAM4_LATENT_LAYOUT_KEY: i4c.IDEOGRAM4_LATENT_LAYOUT_GRID_CHW,
            i4c.IDEOGRAM4_LATENT_SPACE_KEY: i4c.IDEOGRAM4_LATENT_SPACE_DIT_TOKENS,
        },
    )
    return str(path)


def test_trainer_architecture_names():
    trainer = Ideogram4NetworkTrainer()
    assert trainer.architecture == "i4"
    assert trainer.architecture_full_name == "ideogram4"


def test_trainer_rejects_use_mask_loss():
    # Backstop: the mask-loss guard still fires inside process_batch (no mask cache exists for Ideogram yet).
    trainer = Ideogram4NetworkTrainer()
    args = SimpleNamespace(use_mask_loss=True)
    with pytest.raises(ValueError, match="use_mask_loss"):
        trainer.process_batch(args, None, None, None, None, None, None, None, None, None, None, None)


def test_handle_model_specific_args_rejects_use_mask_loss_at_setup():
    # Authoritative fail-fast: rejection happens at setup (handle_model_specific_args), before the model load.
    trainer = Ideogram4NetworkTrainer()
    with pytest.raises(ValueError, match="use_mask_loss"):
        trainer.handle_model_specific_args(SimpleNamespace(use_mask_loss=True))


def test_handle_model_specific_args_rejects_gradient_checkpointing():
    trainer = Ideogram4NetworkTrainer()
    with pytest.raises(ValueError, match="gradient_checkpointing"):
        trainer.handle_model_specific_args(SimpleNamespace(gradient_checkpointing=True))


def test_handle_model_specific_args_rejects_blocks_to_swap():
    trainer = Ideogram4NetworkTrainer()
    with pytest.raises(ValueError, match="blocks_to_swap"):
        trainer.handle_model_specific_args(SimpleNamespace(blocks_to_swap=4))


def test_handle_model_specific_args_defaults_omitted_mixed_precision_to_bf16():
    # Omitted --mixed_precision (None) must default to bf16, not fp32 (which would OOM the 8B DiT). The set
    # value drives BOTH self.dit_dtype here and the base loop's dit_dtype (read later from args.mixed_precision).
    trainer = Ideogram4NetworkTrainer()
    args = SimpleNamespace(mixed_precision=None)
    trainer.handle_model_specific_args(args)
    assert args.mixed_precision == "bf16"
    assert trainer.dit_dtype == torch.bfloat16
    assert args.dit_dtype == "bfloat16"


def test_handle_model_specific_args_respects_explicit_mixed_precision():
    # An explicit choice is never overridden by the bf16 default.
    trainer = Ideogram4NetworkTrainer()
    args = SimpleNamespace(mixed_precision="fp16")
    trainer.handle_model_specific_args(args)
    assert args.mixed_precision == "fp16"
    assert trainer.dit_dtype == torch.float16


def test_neutralize_unused_fp8_args_clears_and_reports():
    # fp8 flags are no-ops for Ideogram; neutralize to False so the base "fp8_scaled requires fp8_base" assert
    # can't abort on them. Returns True iff either had been set.
    a = SimpleNamespace(fp8_base=False, fp8_scaled=True)
    assert neutralize_unused_fp8_args(a) is True
    assert a.fp8_base is False and a.fp8_scaled is False

    b = SimpleNamespace(fp8_base=True, fp8_scaled=False)
    assert neutralize_unused_fp8_args(b) is True
    assert b.fp8_base is False and b.fp8_scaled is False

    c = SimpleNamespace(fp8_base=False, fp8_scaled=False)
    assert neutralize_unused_fp8_args(c) is False
    assert c.fp8_base is False and c.fp8_scaled is False


def test_trainer_call_dit_not_used():
    trainer = Ideogram4NetworkTrainer()
    with pytest.raises(NotImplementedError, match="process_batch"):
        trainer.call_dit()


def test_setup_parser_adds_timestep_args():
    import argparse

    parser = ideogram4_setup_parser(argparse.ArgumentParser())
    args = parser.parse_args(["--ideogram4_timestep_mu", "0.5", "--ideogram4_timestep_std", "1.5"])
    assert args.ideogram4_timestep_mu == 0.5 and args.ideogram4_timestep_std == 1.5


def test_preflight_wiring_aborts_on_stale_cache(tmp_path):
    # The wiring (not just the preflight fn): _build_dataset's loop must propagate the abort BEFORE model load.
    trainer = Ideogram4NetworkTrainer()
    stale = _write_stale_cache(tmp_path / "img_0064x0064_i4.safetensors")
    group = _FakeGroup([_FakeDataset(str(tmp_path), [stale])])
    with pytest.raises(ValueError, match="must set"):
        trainer._preflight_latent_caches(group)


def test_preflight_wiring_passes_and_counts_valid_cache(tmp_path):
    trainer = Ideogram4NetworkTrainer()
    good = _write_valid_cache(tmp_path / "img_0064x0064_i4.safetensors")
    group = _FakeGroup([_FakeDataset(str(tmp_path), [good])])
    assert trainer._preflight_latent_caches(group) == 1


def test_preflight_wiring_skips_datasets_without_cache_dir():
    # cache_directory=None must be skipped cleanly (no os.path.join(None, ...) TypeError), not crash.
    trainer = Ideogram4NetworkTrainer()
    group = _FakeGroup([_FakeDataset(None, ["never-read.safetensors"])])
    assert trainer._preflight_latent_caches(group) == 0


def test_preflight_wiring_dedups_repeated_paths(tmp_path):
    # The same file referenced by two subsets is validated once (dedup), so the count reflects distinct files.
    trainer = Ideogram4NetworkTrainer()
    good = _write_valid_cache(tmp_path / "img_0064x0064_i4.safetensors")
    group = _FakeGroup([_FakeDataset(str(tmp_path), [good]), _FakeDataset(str(tmp_path), [good])])
    assert trainer._preflight_latent_caches(group) == 1
