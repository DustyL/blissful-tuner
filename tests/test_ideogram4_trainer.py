import os
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from safetensors.torch import save_file

from musubi_tuner.ideogram4 import constants as i4c
from musubi_tuner.ideogram4.ideogram4_utils import detect_fp8_scale_layout
from musubi_tuner.ideogram4_train_network import (
    Ideogram4NetworkTrainer,
    ideogram4_setup_parser,
    neutralize_unused_fp8_args,
)
from musubi_tuner.training import trainer_base


class _FakeDataset:
    def __init__(self, cache_directory, files, architecture="i4"):
        self.cache_directory = cache_directory
        self.architecture = architecture
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


def _write_te_cache(cache_dir, item_key="img", arch="i4") -> str:
    # Preflight only validates latent caches that have a paired TE cache (it mirrors the dataset's active set);
    # this makes a latent cache "active" so the wiring tests actually reach the preflight validation.
    path = os.path.join(str(cache_dir), f"{item_key}_{arch}_te.safetensors")
    save_file({"x": torch.zeros(1)}, path)
    return path


def _module_with_scales(*shapes) -> nn.Module:
    root = nn.Module()
    for i, shape in enumerate(shapes):
        lin = nn.Linear(2, 2)
        if shape is not None:
            lin.scale_weight = torch.ones(*shape)
        root.add_module(f"lin{i}", lin)
    return root


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


def test_handle_model_specific_args_rejects_compile():
    # --compile has no compile_transformer hook on Ideogram; the base default raises NotImplementedError only
    # AFTER accelerator.prepare (a confusing late crash). Reject it fail-fast at setup like the other flags.
    trainer = Ideogram4NetworkTrainer()
    with pytest.raises(ValueError, match="compile"):
        trainer.handle_model_specific_args(SimpleNamespace(compile=True))


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
    _write_te_cache(tmp_path)  # paired TE -> the stale latent is "active", so preflight validates (and aborts)
    group = _FakeGroup([_FakeDataset(str(tmp_path), [stale])])
    with pytest.raises(ValueError, match="must set"):
        trainer._preflight_latent_caches(group)


def test_preflight_wiring_passes_and_counts_valid_cache(tmp_path):
    trainer = Ideogram4NetworkTrainer()
    good = _write_valid_cache(tmp_path / "img_0064x0064_i4.safetensors")
    _write_te_cache(tmp_path)
    group = _FakeGroup([_FakeDataset(str(tmp_path), [good])])
    assert trainer._preflight_latent_caches(group) == 1


def test_preflight_wiring_skips_orphan_latent_without_te(tmp_path):
    # An orphan latent cache (no paired TE) is skipped — training ignores it (image_video_dataset.py:644),
    # so preflight must NOT abort even when the orphan is itself stale/invalid. This is the over-strictness fix.
    trainer = Ideogram4NetworkTrainer()
    stale = _write_stale_cache(tmp_path / "img_0064x0064_i4.safetensors")  # NO TE cache written
    group = _FakeGroup([_FakeDataset(str(tmp_path), [stale])])
    assert trainer._preflight_latent_caches(group) == 0  # skipped, not aborted


def test_preflight_wiring_skips_datasets_without_cache_dir():
    # cache_directory=None must be skipped cleanly (no os.path.join(None, ...) TypeError), not crash.
    trainer = Ideogram4NetworkTrainer()
    group = _FakeGroup([_FakeDataset(None, ["never-read.safetensors"])])
    assert trainer._preflight_latent_caches(group) == 0


def test_preflight_wiring_dedups_repeated_paths(tmp_path):
    # The same file referenced by two subsets is validated once (dedup), so the count reflects distinct files.
    trainer = Ideogram4NetworkTrainer()
    good = _write_valid_cache(tmp_path / "img_0064x0064_i4.safetensors")
    _write_te_cache(tmp_path)
    group = _FakeGroup([_FakeDataset(str(tmp_path), [good]), _FakeDataset(str(tmp_path), [good])])
    assert trainer._preflight_latent_caches(group) == 1


def test_detect_fp8_scale_layout_classifies_geometry():
    # The shim normalizes scales to [1] (per-tensor), [out,1] (per-row), [out,num_blocks,1] (per-block).
    assert detect_fp8_scale_layout(_module_with_scales((8, 1), (8, 1))) == "per_row"
    assert detect_fp8_scale_layout(_module_with_scales((1,), (1,))) == "per_tensor"
    assert detect_fp8_scale_layout(_module_with_scales((4, 3, 1))) == "per_block"
    assert detect_fp8_scale_layout(_module_with_scales(None)) == "none"
    assert detect_fp8_scale_layout(_module_with_scales((8, 1), (1,))).startswith("mixed:")


def test_extra_metadata_records_fp8_provenance(monkeypatch):
    # Counterbalance the base ss_fp8_base=False: record the real prequantized-fp8 provenance + scale layout.
    monkeypatch.setattr(trainer_base.NetworkTrainer, "extra_metadata", lambda self, args: {})
    trainer = Ideogram4NetworkTrainer()
    trainer._ideogram4_fp8_layout = "per_row"
    md = trainer.extra_metadata(SimpleNamespace(dit="/models/FLUX-Ideogram-fp8.safetensors"))
    assert md["ss_ideogram4_prequantized_fp8"] is True
    assert md["ss_ideogram4_fp8_source"] == "FLUX-Ideogram-fp8.safetensors"
    assert md["ss_ideogram4_fp8_scale_layout"] == "per_row"


def test_extra_metadata_omits_layout_when_unknown(monkeypatch):
    # If load_transformer never ran (no layout stashed) the layout key is omitted, not emitted as None/"".
    monkeypatch.setattr(trainer_base.NetworkTrainer, "extra_metadata", lambda self, args: {})
    trainer = Ideogram4NetworkTrainer()
    md = trainer.extra_metadata(SimpleNamespace(dit=None))
    assert md["ss_ideogram4_prequantized_fp8"] is True
    assert md["ss_ideogram4_fp8_source"] == ""
    assert "ss_ideogram4_fp8_scale_layout" not in md
