"""JSONL datasources resolve relative media paths: working directory first, then the JSONL's directory."""

import json
import os

import pytest
from PIL import Image

from musubi_tuner.dataset.datasources import (
    ImageJsonlDatasource,
    VideoJsonlDatasource,
    _is_numbered_variant,
    _resolve_jsonl_relative_paths,
)


def _touch_png(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    Image.new("RGB", (8, 8)).save(path)


def _write_jsonl(path, records):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def test_numbered_variant_matching():
    assert _is_numbered_variant("image_path", "image_path")
    assert _is_numbered_variant("image_path_0", "image_path")
    assert _is_numbered_variant("control_path_12", "control_path")
    assert not _is_numbered_variant("image_path_a", "image_path")
    assert not _is_numbered_variant("image_paths", "image_path")
    assert not _is_numbered_variant("caption", "image_path")


def test_jsonl_directory_fallback_and_numbered_keys(tmp_path, monkeypatch):
    # media lives next to the JSONL; cwd is somewhere unrelated
    data_dir = tmp_path / "dataset"
    _touch_png(str(data_dir / "img" / "a.png"))
    _touch_png(str(data_dir / "ctrl" / "a0.png"))
    _touch_png(str(data_dir / "ctrl" / "a1.png"))
    jsonl = data_dir / "train.jsonl"
    _write_jsonl(
        str(jsonl),
        [{"image_path": "img/a.png", "control_path_0": "ctrl/a0.png", "control_path_1": "ctrl/a1.png", "caption": "x"}],
    )
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    ds = ImageJsonlDatasource(str(jsonl))
    rec = ds.data[0]
    assert rec["image_path"] == str(data_dir / "img" / "a.png")
    assert rec["control_path_0"] == str(data_dir / "ctrl" / "a0.png")
    assert rec["control_path_1"] == str(data_dir / "ctrl" / "a1.png")
    assert rec["caption"] == "x"  # non-path keys untouched
    assert os.path.isabs(rec["image_path"])


def test_working_directory_wins_and_is_made_absolute(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    _touch_png(str(cwd / "img" / "a.png"))
    jsonl = tmp_path / "cfg" / "train.jsonl"
    _write_jsonl(str(jsonl), [{"image_path": "img/a.png", "caption": "x"}])
    monkeypatch.chdir(cwd)

    ds = ImageJsonlDatasource(str(jsonl))
    assert ds.data[0]["image_path"] == str(cwd / "img" / "a.png")


def test_both_locations_exist_prefers_cwd_and_warns(tmp_path, monkeypatch, caplog):
    cwd = tmp_path / "cwd"
    _touch_png(str(cwd / "a.png"))
    data_dir = tmp_path / "dataset"
    _touch_png(str(data_dir / "a.png"))
    monkeypatch.chdir(cwd)

    records = [{"image_path": "a.png"}]
    with caplog.at_level("WARNING"):
        _resolve_jsonl_relative_paths(records, str(data_dir / "t.jsonl"), ("image_path",))
    assert records[0]["image_path"] == str(cwd / "a.png")
    assert any("exists both relative to the working directory" in m for m in caplog.messages)


def test_absolute_and_unresolvable_paths_untouched(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    abs_path = str(tmp_path / "abs.png")
    _touch_png(abs_path)
    records = [{"image_path": abs_path}, {"image_path": "missing/nowhere.png"}, {"image_path": ""}]
    _resolve_jsonl_relative_paths(records, str(tmp_path / "t.jsonl"), ("image_path",))
    assert records[0]["image_path"] == abs_path
    assert records[1]["image_path"] == "missing/nowhere.png"  # left as written so the eventual error names it
    assert records[2]["image_path"] == ""


def test_video_jsonl_resolves_video_and_control(tmp_path, monkeypatch):
    data_dir = tmp_path / "dataset"
    (data_dir / "vid").mkdir(parents=True)
    (data_dir / "vid" / "a.mp4").write_bytes(b"")
    (data_dir / "vid" / "a_ctrl.mp4").write_bytes(b"")
    jsonl = data_dir / "train.jsonl"
    _write_jsonl(str(jsonl), [{"video_path": "vid/a.mp4", "control_path": "vid/a_ctrl.mp4", "caption": "x"}])
    monkeypatch.chdir(tmp_path)

    ds = VideoJsonlDatasource(str(jsonl))
    assert ds.data[0]["video_path"] == str(data_dir / "vid" / "a.mp4")
    assert ds.data[0]["control_path"] == str(data_dir / "vid" / "a_ctrl.mp4")


if __name__ == "__main__":
    pytest.main([__file__])
