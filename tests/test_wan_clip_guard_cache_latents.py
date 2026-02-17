"""Tests for WAN cache-latents CLIP guard (LC-03).

WAN 2.2 A14B tasks do not use CLIP. Passing --clip should be rejected when
the task is known (via --task).
"""

import argparse
import unittest


class TestWanCacheLatentsClipGuard(unittest.TestCase):
    def test_clip_rejected_for_wan22_a14b_tasks(self) -> None:
        from musubi_tuner.wan_cache_latents import validate_wan_cache_latents_args

        args = argparse.Namespace(task="i2v-A14B", clip="/tmp/clip.pth")
        with self.assertRaises(ValueError) as ctx:
            validate_wan_cache_latents_args(args)
        self.assertIn("WAN 2.2", str(ctx.exception))

    def test_clip_allowed_for_wan21_i2v_tasks(self) -> None:
        from musubi_tuner.wan_cache_latents import validate_wan_cache_latents_args

        args = argparse.Namespace(task="i2v-14B", clip="/tmp/clip.pth")
        validate_wan_cache_latents_args(args)  # should not raise


if __name__ == "__main__":
    unittest.main()

