import unittest
from pathlib import Path


class TestCacheMaskWiring(unittest.TestCase):
    def test_cache_parser_exposes_cache_mask_args(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        text = (repo_root / "src/musubi_tuner/cache_latents.py").read_text(encoding="utf-8")
        self.assertIn("--cache_mask_gamma", text)
        self.assertIn("--cache_mask_min_weight", text)

    def test_cache_scripts_apply_cache_mask_transforms_before_downsample(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        targets = [
            repo_root / "src/musubi_tuner/wan_cache_latents.py",
            repo_root / "src/musubi_tuner/qwen_image_cache_latents.py",
            repo_root / "src/musubi_tuner/flux_2_cache_latents.py",
            repo_root / "src/musubi_tuner/zimage_cache_latents.py",
        ]

        for path in targets:
            text = path.read_text(encoding="utf-8")
            self.assertIn(
                "apply_cache_mask_transforms",
                text,
                msg=f"Expected cache-time mask transforms to be wired in {path}",
            )


if __name__ == "__main__":
    unittest.main()
