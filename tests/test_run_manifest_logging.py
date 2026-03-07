from __future__ import annotations

import argparse
import json
from pathlib import Path

from musubi_tuner.utils import train_utils


def test_write_run_manifest_snapshots_configs_and_env(tmp_path, monkeypatch):
    training_config = tmp_path / "train_config.toml"
    dataset_config = tmp_path / "dataset.toml"
    sample_prompts = tmp_path / "sample_prompts.txt"
    tracker_config = tmp_path / "tracker.toml"
    launcher_script = tmp_path / "train_naomi_wan22.sh"
    launcher_env = tmp_path / "env_wan22.sh"
    invocation_note = tmp_path / "invocation_note.txt"
    output_dir = tmp_path / "output"
    run_dir = tmp_path / "logs" / "run123"

    training_config.write_text("[training]\nmax_train_steps = 123\n", encoding="utf-8")
    dataset_config.write_text("[datasets]\n", encoding="utf-8")
    sample_prompts.write_text("NAOMI portrait\n", encoding="utf-8")
    tracker_config.write_text("[wandb]\nproject = 'naomi-tests'\n", encoding="utf-8")
    launcher_script.write_text("#!/bin/bash\necho run\n", encoding="utf-8")
    launcher_env.write_text("export OMP_NUM_THREADS=16\n", encoding="utf-8")
    invocation_note.write_text("config=/tmp/train.toml\n", encoding="utf-8")

    monkeypatch.setenv("CACHE_MASK_GAMMA", "0.7")
    monkeypatch.setenv("OMP_NUM_THREADS", "16")
    monkeypatch.setenv("RUN_MANIFEST_LAUNCHER_SCRIPT", str(launcher_script))
    monkeypatch.setenv("RUN_MANIFEST_ENV_SCRIPT", str(launcher_env))
    monkeypatch.setenv("RUN_MANIFEST_INVOCATION_NOTE", str(invocation_note))

    args = argparse.Namespace(
        output_dir=str(output_dir),
        config_file=str(training_config.with_suffix("")),
        dataset_config=str(dataset_config),
        sample_prompts=str(sample_prompts),
        log_tracker_config=str(tracker_config),
        wandb_api_key="secret",
        huggingface_token="also-secret",
        some_list=["a", "b"],
        some_flag=True,
    )

    created = train_utils.write_run_manifest(args, run_dir=str(run_dir), tracker_name="network_train")

    manifest_path = Path(created["manifest_json"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["tracker_name"] == "network_train"
    assert manifest["args"]["config_file"] == str(training_config.with_suffix(""))
    assert manifest["args"]["some_list"] == ["a", "b"]
    assert manifest["env"]["CACHE_MASK_GAMMA"] == "0.7"
    assert "wandb_api_key" not in manifest["args"]
    assert "huggingface_token" not in manifest["args"]

    snapshot_dir = Path(created["snapshot_dir"])
    assert (snapshot_dir / "training_config.toml").read_text(encoding="utf-8") == training_config.read_text(encoding="utf-8")
    assert (snapshot_dir / "dataset_config.toml").read_text(encoding="utf-8") == dataset_config.read_text(encoding="utf-8")
    assert (snapshot_dir / "sample_prompts.txt").read_text(encoding="utf-8") == sample_prompts.read_text(encoding="utf-8")
    assert (snapshot_dir / "tracker_config.toml").read_text(encoding="utf-8") == tracker_config.read_text(encoding="utf-8")
    assert (snapshot_dir / "launcher_script.sh").read_text(encoding="utf-8") == launcher_script.read_text(encoding="utf-8")
    assert (snapshot_dir / "launcher_env.sh").read_text(encoding="utf-8") == launcher_env.read_text(encoding="utf-8")
    assert (snapshot_dir / "invocation_note.txt").read_text(encoding="utf-8") == invocation_note.read_text(encoding="utf-8")

    mirror_manifest = output_dir / "run_manifests" / "run123" / "run_manifest.json"
    assert mirror_manifest.exists()
