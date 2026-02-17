from __future__ import annotations

from unittest import mock

from musubi_tuner.flux_2 import flux2_utils
from musubi_tuner.zimage import zimage_utils


def test_load_mistral3_processor_tries_local_path_first(tmp_path):
    te_dir = tmp_path / "te"
    te_dir.mkdir()

    sentinel = object()

    def fake_from_pretrained(model_id, *args, **kwargs):
        if model_id == str(te_dir):
            raise OSError("missing tokenizer files")
        if model_id == flux2_utils.M3_TOKENIZER_ID:
            return sentinel
        raise AssertionError(f"Unexpected from_pretrained({model_id!r})")

    with mock.patch.object(flux2_utils.AutoProcessor, "from_pretrained", side_effect=fake_from_pretrained) as patched:
        proc = flux2_utils.load_mistral3_processor(str(te_dir))

    assert proc is sentinel
    assert patched.call_args_list[0].args[0] == str(te_dir)
    assert patched.call_args_list[1].args[0] == flux2_utils.M3_TOKENIZER_ID


def test_load_qwen2_tokenizer_local_first_tries_ckpt_path_before_hf_id(tmp_path):
    te_dir = tmp_path / "te"
    te_dir.mkdir()

    sentinel = object()
    tokenizer_id = "Qwen/Qwen3-8B"

    def fake_from_pretrained(model_id, *args, **kwargs):
        if model_id == str(te_dir):
            raise OSError("missing tokenizer files")
        if model_id == tokenizer_id:
            return sentinel
        raise AssertionError(f"Unexpected from_pretrained({model_id!r})")

    with mock.patch.object(zimage_utils.Qwen2Tokenizer, "from_pretrained", side_effect=fake_from_pretrained) as patched:
        tok = zimage_utils.load_qwen2_tokenizer_local_first(str(te_dir), tokenizer_id=tokenizer_id, subfolder=None)

    assert tok is sentinel
    assert patched.call_args_list[0].args[0] == str(te_dir)
    assert patched.call_args_list[1].args[0] == tokenizer_id


def test_load_qwen2_tokenizer_local_first_skips_local_attempt_for_non_dir_path():
    sentinel = object()
    tokenizer_id = "Qwen/Qwen3-4B"

    with mock.patch.object(zimage_utils.Qwen2Tokenizer, "from_pretrained", return_value=sentinel) as patched:
        tok = zimage_utils.load_qwen2_tokenizer_local_first("/not/a/dir", tokenizer_id=tokenizer_id, subfolder=None)

    assert tok is sentinel
    assert patched.call_args_list[0].args[0] == tokenizer_id
