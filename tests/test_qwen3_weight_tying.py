import unittest
from unittest.mock import patch

import torch

from musubi_tuner.zimage import zimage_utils


class _DummyQwen3:
    def __init__(self):
        self.loaded_sd = None

    def load_state_dict(self, sd, strict=True, assign=True):  # noqa: ARG002
        self.loaded_sd = sd
        return {"ok": True}

    def to(self, *args, **kwargs):  # noqa: ARG002
        return self


class TestQwen3WeightTying(unittest.TestCase):
    def _call_load_qwen3(self, state_dict, *, is_8b: bool):
        dummy_model = _DummyQwen3()
        with (
            patch.object(zimage_utils.Qwen3ForCausalLM, "_from_config", return_value=dummy_model),
            patch.object(zimage_utils.Qwen2Tokenizer, "from_pretrained", return_value=object()),
        ):
            tokenizer, model = zimage_utils.load_qwen3(
                ckpt_path="unused",
                dtype=None,
                device="cpu",
                disable_mmap=True,
                state_dict=state_dict,
                is_8b=is_8b,
                tokenizer_id="unused",
            )
        self.assertIsNotNone(tokenizer)
        self.assertIs(model, dummy_model)
        return dummy_model

    def test_qwen3_4b_forces_tied_lm_head(self):
        embed = torch.randn(2, 3)
        lm_head = torch.randn(2, 3)  # distinct tensor
        sd = {"model.embed_tokens.weight": embed, "lm_head.weight": lm_head}

        model = self._call_load_qwen3(sd, is_8b=False)
        self.assertIs(model.loaded_sd["lm_head.weight"], embed)

    def test_qwen3_8b_does_not_overwrite_lm_head(self):
        embed = torch.randn(2, 3)
        lm_head = torch.randn(2, 3)  # distinct tensor
        sd = {"model.embed_tokens.weight": embed, "lm_head.weight": lm_head}

        model = self._call_load_qwen3(sd, is_8b=True)
        self.assertIs(model.loaded_sd["lm_head.weight"], lm_head)

    def test_qwen3_8b_fallback_when_lm_head_missing(self):
        embed = torch.randn(2, 3)
        sd = {"model.embed_tokens.weight": embed}

        model = self._call_load_qwen3(sd, is_8b=True)
        self.assertIn("lm_head.weight", model.loaded_sd)
        self.assertIs(model.loaded_sd["lm_head.weight"], embed)


if __name__ == "__main__":
    unittest.main()
