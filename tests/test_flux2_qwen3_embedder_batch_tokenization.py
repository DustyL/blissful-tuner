from __future__ import annotations

from types import SimpleNamespace

import torch

from musubi_tuner.flux_2 import flux2_utils


def test_qwen3_embedder_batches_tokenization_calls_tokenizer_once():
    class FakeTokenizer:
        def __init__(self):
            self.calls = []
            self.template_calls = []

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
            self.template_calls.append((messages, tokenize, add_generation_prompt, enable_thinking))
            return f"TEMPLATE:{messages[0]['content']}"

        def __call__(self, text, **kwargs):
            self.calls.append(text)
            batch_size = 1 if isinstance(text, str) else len(text)
            seq_len = 4
            return {
                "input_ids": torch.zeros((batch_size, seq_len), dtype=torch.int64),
                "attention_mask": torch.ones((batch_size, seq_len), dtype=torch.int64),
            }

    class FakeModel:
        def __init__(self):
            self.device = torch.device("cpu")
            self.dtype = torch.bfloat16

        def __call__(self, input_ids, attention_mask, output_hidden_states, use_cache):
            bsz, seq_len = input_ids.shape
            hidden = 2
            hidden_states = [torch.zeros((bsz, seq_len, hidden), dtype=torch.float32) for _ in range(28)]
            return SimpleNamespace(hidden_states=hidden_states)

    tok = FakeTokenizer()
    model = FakeModel()
    embedder = flux2_utils.Qwen3Embedder(tok, model)

    out = embedder(["a", "b"])
    assert out.shape == (2, 4, 6)  # B, L, len(OUTPUT_LAYERS_QWEN3)*hidden
    assert len(tok.calls) == 1, f"Expected tokenizer to be called once, got {len(tok.calls)} calls: {tok.calls}"
    assert len(tok.template_calls) == 2, f"Expected apply_chat_template per prompt, got {len(tok.template_calls)}"


def test_qwen3_embedder_accepts_single_string():
    class FakeTokenizer:
        def __init__(self):
            self.calls = []

        def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
            return "TEMPLATE"

        def __call__(self, text, **kwargs):
            self.calls.append(text)
            return {
                "input_ids": torch.zeros((1, 3), dtype=torch.int64),
                "attention_mask": torch.ones((1, 3), dtype=torch.int64),
            }

    class FakeModel:
        def __init__(self):
            self.device = torch.device("cpu")
            self.dtype = torch.bfloat16

        def __call__(self, input_ids, attention_mask, output_hidden_states, use_cache):
            bsz, seq_len = input_ids.shape
            hidden = 2
            hidden_states = [torch.zeros((bsz, seq_len, hidden), dtype=torch.float32) for _ in range(28)]
            return SimpleNamespace(hidden_states=hidden_states)

    tok = FakeTokenizer()
    model = FakeModel()
    embedder = flux2_utils.Qwen3Embedder(tok, model)

    out = embedder("a")
    assert out.shape == (1, 3, 6)
    assert len(tok.calls) == 1
