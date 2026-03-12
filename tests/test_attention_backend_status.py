"""Tests for the pure attention backend status helper."""

import unittest


class TestGetAttentionBackendStatus(unittest.TestCase):
    def test_returns_dict_with_expected_keys(self):
        from musubi_tuner.hunyuan_model.attention import get_attention_backend_status

        status = get_attention_backend_status()
        self.assertIsInstance(status, dict)
        expected_keys = {"Flash Attention", "Sage Attention", "Xformers", "CuTE"}
        self.assertEqual(set(status.keys()), expected_keys)

    def test_values_are_bools(self):
        from musubi_tuner.hunyuan_model.attention import get_attention_backend_status

        status = get_attention_backend_status()
        for name, available in status.items():
            self.assertIsInstance(available, bool, f"{name} should be bool, got {type(available)}")

    def test_idempotent_same_result(self):
        from musubi_tuner.hunyuan_model.attention import get_attention_backend_status

        first = get_attention_backend_status()
        second = get_attention_backend_status()
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
