"""Tests for tools/lora_eval/inventory.py — the LoRA inventory parser.

Pins the parser's behavior on the heterogeneous naming conventions observed
in the live `~/SwarmUI/Models/Lora/loras/` and `~/SwarmUI/Models/Lora/WAN/`
trees. The parser is the load-bearing decision-making logic for the entire
LoRA eval pipeline — silent mis-grouping at this layer would propagate to
GPU-time eval and ultimately to deletion decisions, so the regex behaviors
here are pinned independently.

Fixture cases cover (a) every observed naming variant, (b) every parser fix
that landed in response to live-data issues (named-tag over-peel, K-shorthand,
DiffusionPipe path parsing, hyphen-EMA, dash-noise), and (c) the post-pass
collision/sibling/cross-root detectors.
"""

from __future__ import annotations

import json
import struct
import sys
import unittest
from pathlib import Path

# Add repo root + tools/ subdirectory to sys.path so `tools.lora_eval.inventory`
# is importable without requiring the project to be installed as a package.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from tools.lora_eval.inventory import (  # noqa: E402
    DEFAULT_TRIGGERS,
    attach_fallback_partners,
    collect_entries,
    detect_trigger,
    find_cross_root_duplicates,
    group_into_lineages,
    parse_diffusion_pipe_path,
    parse_filename,
    read_safetensors_metadata,
)


class TestParseFilename(unittest.TestCase):
    """Exact-match expectations for parsed lineage_basename + suffix fields."""

    def test_kohya_step_and_noise(self) -> None:
        p = parse_filename("WAN2.2-LoRA-V3_000006000_high_noise.safetensors")
        self.assertEqual(p.lineage_basename, "WAN2.2-LoRA-V3")
        self.assertEqual(p.step, 6000)
        self.assertEqual(p.noise, "high")
        self.assertIsNone(p.ema_tag)
        self.assertFalse(p.is_final)

    def test_kohya_final_with_noise_dash_marker(self) -> None:
        p = parse_filename("ELLE_wan22_h200_character_high_noise-final.safetensors")
        self.assertEqual(p.lineage_basename, "ELLE_wan22_h200_character")
        self.assertIsNone(p.step)
        self.assertEqual(p.noise, "high")
        self.assertTrue(p.is_final)

    def test_step_prefixed_form(self) -> None:
        p = parse_filename("dlay_wan22_persona_lokr-v1-step00007000.safetensors")
        self.assertEqual(p.lineage_basename, "dlay_wan22_persona_lokr-v1")
        self.assertEqual(p.step, 7000)
        self.assertIsNone(p.noise)

    def test_k_shorthand_step_merges_with_explicit_step(self) -> None:
        # `-13K-steps` should produce the same lineage as `-step00006240`.
        p1 = parse_filename("dlay_man_low_noise_lokr_v2-13K-steps.safetensors")
        p2 = parse_filename("dlay_man_low_noise_lokr_v2-step00006240.safetensors")
        self.assertEqual(p1.lineage_basename, p2.lineage_basename)
        self.assertEqual(p1.step, 13000)
        self.assertEqual(p2.step, 6240)

    def test_ema_underscore_tag(self) -> None:
        p = parse_filename("crln_persona_lora_ema_015.safetensors")
        self.assertEqual(p.lineage_basename, "crln_persona_lora")
        self.assertEqual(p.ema_tag, "015")

    def test_ema_hyphen_tag(self) -> None:
        # Live fixture: carlen_lora_ema-sigma_balanced_0-2.safetensors uses a
        # dash separator between `ema` and the tag.
        p = parse_filename("carlen_lora_ema-sigma_balanced_0-2.safetensors")
        self.assertEqual(p.lineage_basename, "carlen_lora")
        self.assertEqual(p.ema_tag, "sigma_balanced_0-2")

    def test_bare_ema_tag_is_empty_string(self) -> None:
        p = parse_filename("crln_persona_lora_ema.safetensors")
        self.assertEqual(p.lineage_basename, "crln_persona_lora")
        self.assertEqual(p.ema_tag, "")

    def test_named_tag_in_basename_NOT_peeled(self) -> None:
        # `WAN2.2-LoRA-V3` and `WAN2.2-T2V-Personal-LoRA` had a regression
        # where a greedy named-tag peel mangled them to lineage `WAN2.2`. The
        # current parser keeps the dashed basename intact.
        p = parse_filename("WAN2.2-T2V-Personal-LoRA_000005500_high_noise.safetensors")
        self.assertEqual(p.lineage_basename, "WAN2.2-T2V-Personal-LoRA")
        self.assertEqual(p.step, 5500)
        self.assertEqual(p.noise, "high")

    def test_mid_string_HIGH_noise_is_NOT_a_suffix(self) -> None:
        # `dlay_man_HIGH_noise_lokr_v3-musubi` deliberately has `HIGH_noise`
        # mid-string. The end-anchored noise regex must not peel it.
        p = parse_filename("dlay_man_HIGH_noise_lokr_v3-musubi-step00009750.safetensors")
        self.assertEqual(p.lineage_basename, "dlay_man_HIGH_noise_lokr_v3-musubi")
        self.assertEqual(p.step, 9750)
        self.assertIsNone(p.noise)

    def test_underscore_final_in_basename_NOT_treated_as_final(self) -> None:
        # `dlay_wan22_gh200_character_final` has `_final` mid-name; only `-final`
        # (with hyphen) is the canonical "final variant" marker.
        p = parse_filename("dlay_wan22_gh200_character_final_000003750_low_noise.safetensors")
        self.assertEqual(p.lineage_basename, "dlay_wan22_gh200_character_final")
        self.assertEqual(p.step, 3750)
        self.assertEqual(p.noise, "low")
        self.assertFalse(p.is_final)

    def test_dash_noise_form(self) -> None:
        # DiffusionPipe-style dirnames use dashes throughout (`wan22-high-noise`).
        # The noise regex accepts both `_` and `-` separators.
        p = parse_filename("wan22-high-noise.safetensors")
        self.assertEqual(p.lineage_basename, "wan22")
        self.assertEqual(p.noise, "high")


class TestParseDiffusionPipePath(unittest.TestCase):
    """Path-based parsing for the `<dir>/[<timestamp>/]epoch<N>/adapter_model.safetensors` shape."""

    def setUp(self) -> None:
        self.root = Path("/fake/scan/root")

    def _path(self, *parts: str) -> Path:
        return self.root.joinpath(*parts)

    def test_basic_layout(self) -> None:
        p = parse_diffusion_pipe_path(
            self._path("wan2.2_dlay_lora_exp2_high_noise", "epoch90", "adapter_model.safetensors"),
            self.root,
        )
        assert p is not None
        self.assertEqual(p.lineage_basename, "wan2.2_dlay_lora_exp2")
        self.assertEqual(p.step, 90)
        self.assertEqual(p.noise, "high")

    def test_with_timestamp_dir(self) -> None:
        # Timestamp gets stitched into the lineage name with `@` so concurrent
        # runs of the same training config stay as separate lineages.
        p = parse_diffusion_pipe_path(
            self._path("config_low_noise", "20251103_20-26-25", "epoch20", "adapter_model.safetensors"),
            self.root,
        )
        assert p is not None
        self.assertEqual(p.lineage_basename, "config@20251103_20-26-25")
        self.assertEqual(p.step, 20)
        self.assertEqual(p.noise, "low")

    def test_epoch_baked_into_dirname(self) -> None:
        # `wan22-low-noise-epoch300/adapter_model.safetensors` — epoch is in the
        # parent dir name, no nested `epochN/` subdir.
        p = parse_diffusion_pipe_path(
            self._path("wan22-low-noise-epoch300", "adapter_model.safetensors"),
            self.root,
        )
        assert p is not None
        self.assertEqual(p.lineage_basename, "wan22")
        self.assertEqual(p.step, 300)
        self.assertEqual(p.noise, "low")

    def test_non_diffusion_pipe_filename_returns_none(self) -> None:
        # Different filename: not DiffusionPipe.
        self.assertIsNone(parse_diffusion_pipe_path(self._path("anything", "epoch50", "model.safetensors"), self.root))

    def test_bare_adapter_model_at_root_returns_none(self) -> None:
        # No lineage info to extract — falls through to the orphan path.
        self.assertIsNone(parse_diffusion_pipe_path(self._path("adapter_model.safetensors"), self.root))

    def test_nested_outer_campaign_dirs_are_ignored(self) -> None:
        # Outer dirs above the lineage_dir act as "campaign labels" and are
        # not currently part of the lineage. Pin this behavior so a future
        # change to include them is a deliberate decision.
        p = parse_diffusion_pipe_path(
            self._path("WAN2.2-Unknown-LoRA", "WAN2.2-High-Noise", "wan22-high-noise-epoch150", "adapter_model.safetensors"),
            self.root,
        )
        assert p is not None
        self.assertEqual(p.lineage_basename, "wan22")
        self.assertEqual(p.noise, "high")
        self.assertEqual(p.step, 150)


class TestTriggerDetection(unittest.TestCase):
    def test_token_in_filename(self) -> None:
        self.assertEqual(detect_trigger("ELLE_persona_lora.safetensors", DEFAULT_TRIGGERS, "DLAY"), "ELLE")

    def test_token_with_dash_boundaries(self) -> None:
        self.assertEqual(detect_trigger("WAN2.2-T2V-LoRA-DLAY.safetensors", DEFAULT_TRIGGERS, "DLAY"), "DLAY")

    def test_no_token_falls_back_to_default(self) -> None:
        self.assertEqual(detect_trigger("anonymous_lora.safetensors", DEFAULT_TRIGGERS, "DLAY"), "DLAY")

    def test_case_insensitive(self) -> None:
        self.assertEqual(detect_trigger("crln_persona_lora.safetensors", DEFAULT_TRIGGERS, "DLAY"), "CRLN")


class TestGroupingAndPairing(unittest.TestCase):
    """End-to-end: walk a synthetic dir tree, assert grouping invariants."""

    def setUp(self) -> None:
        self.root = self._make_synthetic_tree()

    def tearDown(self) -> None:
        # tmpfs cleanup is left to the OS — pytest won't accumulate enough
        # fixtures here to matter, and we want failures to leave the tree
        # available for inspection.
        pass

    @staticmethod
    def _write_minimal_safetensors(path: Path, *, size_padding: int = 0) -> None:
        """Write a minimal valid safetensors file (empty tensor map) at `path`.

        Layout: 8-byte little-endian header length, then JSON `{}`. This is
        enough for the inventory's size and metadata reads without requiring
        a real LoRA on disk.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        header = b"{}"
        body = struct.pack("<Q", len(header)) + header + b"\x00" * size_padding
        path.write_bytes(body)

    def _make_synthetic_tree(self) -> Path:
        import tempfile

        root = Path(tempfile.mkdtemp(prefix="lora_eval_inventory_test_"))

        # KYLA collision: same basename in two sibling subdirs (production +
        # test). The grouper should split these by parent_dirname.
        for step in (3000, 3200):
            for noise in ("high", "low"):
                self._write_minimal_safetensors(
                    root / "KYLA_persona_B200_test" / f"KYLA_persona_B200_{step:09d}_{noise}_noise.safetensors"
                )
        for step in (6400, 6600):
            for noise in ("high", "low"):
                self._write_minimal_safetensors(
                    root / "KYLA_persona_B200" / f"KYLA_persona_B200_{step:09d}_{noise}_noise.safetensors"
                )

        # DiffusionPipe layout: epoch in subdir, with timestamp.
        for epoch in (10, 20):
            self._write_minimal_safetensors(
                root / "config_low_noise" / "20251103_20-26-25" / f"epoch{epoch}" / "adapter_model.safetensors"
            )

        # Dash-noise pair that should auto-merge after Fix #4.
        for epoch in (200, 300):
            self._write_minimal_safetensors(root / f"wan22-low-noise-epoch{epoch}" / "adapter_model.safetensors")
        self._write_minimal_safetensors(root / "wan22-high-noise-epoch200" / "adapter_model.safetensors")

        # Bare adapter_model at root — should become an orphan (`no_lineage_info`).
        self._write_minimal_safetensors(root / "adapter_model.safetensors")

        # Plain kohya-style file with K-shorthand + numeric step that should merge.
        self._write_minimal_safetensors(root / "demo_lora-13K-steps.safetensors")
        self._write_minimal_safetensors(root / "demo_lora-step00006240.safetensors")

        return root

    def test_kyla_collision_splits_into_two_lineages(self) -> None:
        entries = collect_entries([self.root], triggers=DEFAULT_TRIGGERS, default_trigger="DLAY", read_metadata=False)
        lineages, _orphans = group_into_lineages(entries)
        names = {lin.name for lin in lineages}
        self.assertIn("KYLA_persona_B200", names)
        self.assertIn("KYLA_persona_B200_test", names)
        # And they have the right number of variants in each.
        prod = next(lin for lin in lineages if lin.name == "KYLA_persona_B200")
        test = next(lin for lin in lineages if lin.name == "KYLA_persona_B200_test")
        self.assertEqual(prod.variant_count, 2)
        self.assertEqual(test.variant_count, 2)
        self.assertTrue(all(v.is_paired for v in prod.variants))
        self.assertTrue(all(v.is_paired for v in test.variants))

    def test_dash_noise_pair_auto_merges(self) -> None:
        entries = collect_entries([self.root], triggers=DEFAULT_TRIGGERS, default_trigger="DLAY", read_metadata=False)
        lineages, _orphans = group_into_lineages(entries)
        wan22 = next((lin for lin in lineages if lin.name == "wan22"), None)
        self.assertIsNotNone(wan22, "expected 'wan22' lineage from dash-noise auto-merge")
        # Two epochs (200 paired, 300 unpaired-orphan) so we should see one
        # paired variant for step 200.
        steps = sorted(v.step for v in wan22.variants)  # type: ignore[union-attr]
        self.assertIn(200, steps)
        paired_200 = [v for v in wan22.variants if v.step == 200 and v.is_paired]  # type: ignore[union-attr]
        self.assertEqual(len(paired_200), 1)

    def test_bare_adapter_model_becomes_orphan(self) -> None:
        entries = collect_entries([self.root], triggers=DEFAULT_TRIGGERS, default_trigger="DLAY", read_metadata=False)
        lineages, orphans = group_into_lineages(entries)
        self.assertNotIn("adapter_model", {lin.name for lin in lineages})
        self.assertIn("no_lineage_info", {o.reason for o in orphans})

    def test_paths_in_variants_are_absolute(self) -> None:
        entries = collect_entries([self.root], triggers=DEFAULT_TRIGGERS, default_trigger="DLAY", read_metadata=False)
        lineages, _orphans = group_into_lineages(entries)
        # Every variant's *_path field that's set should be an absolute path.
        for lin in lineages:
            for v in lin.variants:
                for path in (v.high_path, v.low_path, v.single_path):
                    if path is not None:
                        self.assertTrue(Path(path).is_absolute(), f"variant path not absolute: {path}")
                        self.assertTrue(Path(path).exists(), f"variant path missing: {path}")

    def test_k_shorthand_merges_with_step(self) -> None:
        entries = collect_entries([self.root], triggers=DEFAULT_TRIGGERS, default_trigger="DLAY", read_metadata=False)
        lineages, _orphans = group_into_lineages(entries)
        demo = next((lin for lin in lineages if lin.name == "demo_lora"), None)
        self.assertIsNotNone(demo)
        steps = sorted(v.step for v in demo.variants)  # type: ignore[union-attr]
        self.assertEqual(steps, [6240, 13000])


class TestMultiRootAndCrossRootDetection(unittest.TestCase):
    def setUp(self) -> None:
        import tempfile

        # Build TWO synthetic roots that share a lineage name to exercise
        # cross-root duplicate detection.
        self.root_a = Path(tempfile.mkdtemp(prefix="lora_eval_root_a_"))
        self.root_b = Path(tempfile.mkdtemp(prefix="lora_eval_root_b_"))
        TestGroupingAndPairing._write_minimal_safetensors(self.root_a / "shared_lora_000001000_high_noise.safetensors")
        TestGroupingAndPairing._write_minimal_safetensors(self.root_a / "shared_lora_000001000_low_noise.safetensors")
        TestGroupingAndPairing._write_minimal_safetensors(self.root_b / "shared_lora_000001000_high_noise.safetensors")
        TestGroupingAndPairing._write_minimal_safetensors(self.root_b / "shared_lora_000001000_low_noise.safetensors")

    def test_same_name_in_two_roots_yields_two_lineages(self) -> None:
        entries = collect_entries(
            [self.root_a, self.root_b],
            triggers=DEFAULT_TRIGGERS,
            default_trigger="DLAY",
            read_metadata=False,
        )
        lineages, _orphans = group_into_lineages(entries)
        shared = [lin for lin in lineages if lin.name == "shared_lora"]
        self.assertEqual(len(shared), 2)
        self.assertEqual({lin.source_root for lin in shared}, {str(self.root_a), str(self.root_b)})

    def test_cross_root_duplicate_detected(self) -> None:
        entries = collect_entries(
            [self.root_a, self.root_b],
            triggers=DEFAULT_TRIGGERS,
            default_trigger="DLAY",
            read_metadata=False,
        )
        lineages, _orphans = group_into_lineages(entries)
        crd = find_cross_root_duplicates(lineages)
        names = {name for name, _ in crd}
        self.assertIn("shared_lora", names)


class TestFallbackPairing(unittest.TestCase):
    """Strategy 1: nearest-step fallback for unpaired_noise orphans."""

    def setUp(self) -> None:
        import tempfile

        self.root = Path(tempfile.mkdtemp(prefix="lora_eval_fallback_"))
        # Lineage with a complete pair at step 3500 and an unpaired low at step
        # 3000 (no high partner). The fallback should match step 3000 low to
        # the step 3500 high.
        for noise in ("high", "low"):
            TestGroupingAndPairing._write_minimal_safetensors(self.root / f"ELLE_lora_000003500_{noise}_noise.safetensors")
        TestGroupingAndPairing._write_minimal_safetensors(self.root / "ELLE_lora_000003000_low_noise.safetensors")

    def test_fallback_partner_attached_to_unpaired_orphan(self) -> None:
        entries = collect_entries([self.root], triggers=DEFAULT_TRIGGERS, default_trigger="DLAY", read_metadata=False)
        lineages, orphans = group_into_lineages(entries)
        attach_fallback_partners(lineages, orphans, max_step_distance=None)

        unpaired = [o for o in orphans if o.reason == "unpaired_noise"]
        self.assertEqual(len(unpaired), 1)
        o = unpaired[0]
        self.assertEqual(o.file, "ELLE_lora_000003000_low_noise.safetensors")
        self.assertIsNotNone(o.fallback_partner)
        fp = o.fallback_partner
        assert fp is not None
        self.assertEqual(fp["step"], 3500)
        self.assertEqual(fp["channel"], "high")
        self.assertEqual(fp["step_distance"], 500)
        self.assertEqual(fp["source"], "paired_variant")
        self.assertTrue(Path(fp["path"]).is_absolute())

    def test_max_step_distance_skips_too_far_partners(self) -> None:
        entries = collect_entries([self.root], triggers=DEFAULT_TRIGGERS, default_trigger="DLAY", read_metadata=False)
        lineages, orphans = group_into_lineages(entries)
        # Distance is 500; with max=200 we should skip the partner.
        attach_fallback_partners(lineages, orphans, max_step_distance=200)
        unpaired = [o for o in orphans if o.reason == "unpaired_noise"]
        self.assertEqual(len(unpaired), 1)
        self.assertIsNone(unpaired[0].fallback_partner)


class TestDiffusionPipeTimestampMerge(unittest.TestCase):
    """Strategy 2: --merge-diffusion-pipe-timestamps drops the @<timestamp>
    suffix so high/low halves trained in separate jobs collapse into one
    lineage and pair via the existing logic."""

    def setUp(self) -> None:
        import tempfile

        self.root = Path(tempfile.mkdtemp(prefix="lora_eval_dp_merge_"))
        # High noise trained at one timestamp, low noise at another, same epoch.
        TestGroupingAndPairing._write_minimal_safetensors(
            self.root / "wan2.2_dlay_lora_high_noise" / "20251103_10-04-43" / "epoch100" / "adapter_model.safetensors"
        )
        TestGroupingAndPairing._write_minimal_safetensors(
            self.root / "wan2.2_dlay_lora_low_noise" / "20251103_02-14-02" / "epoch100" / "adapter_model.safetensors"
        )

    def test_default_keeps_timestamps_separate(self) -> None:
        entries = collect_entries([self.root], triggers=DEFAULT_TRIGGERS, default_trigger="DLAY", read_metadata=False)
        lineages, orphans = group_into_lineages(entries)
        # Without merging, each timestamp is its own lineage. Because each
        # lineage holds only ONE channel for its single epoch, neither can
        # pair — both files become `unpaired_noise` orphans, each tagged
        # with its own timestamped lineage name in `unpaired_info`.
        self.assertEqual(lineages, [])
        unpaired = [o for o in orphans if o.reason == "unpaired_noise"]
        self.assertEqual(len(unpaired), 2)
        lineage_names = {o.unpaired_info["lineage"] for o in unpaired if o.unpaired_info}
        self.assertEqual(
            lineage_names,
            {"wan2.2_dlay_lora@20251103_02-14-02", "wan2.2_dlay_lora@20251103_10-04-43"},
        )

    def test_merge_flag_collapses_into_one_paired_lineage(self) -> None:
        entries = collect_entries(
            [self.root],
            triggers=DEFAULT_TRIGGERS,
            default_trigger="DLAY",
            read_metadata=False,
            merge_diffusion_pipe_timestamps=True,
        )
        lineages, _orphans = group_into_lineages(entries)
        self.assertEqual(len(lineages), 1)
        lin = lineages[0]
        self.assertEqual(lin.name, "wan2.2_dlay_lora")
        # Single variant at epoch 100, paired (both halves present).
        self.assertEqual(lin.variant_count, 1)
        self.assertTrue(lin.variants[0].is_paired)
        self.assertEqual(lin.variants[0].step, 100)


class TestSafetensorsHeaderReader(unittest.TestCase):
    def test_minimal_header_returns_empty_dict(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            header = b"{}"
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
            path = Path(f.name)
        self.assertEqual(read_safetensors_metadata(path), {})

    def test_metadata_with_known_keys_filtered(self) -> None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            header_obj = {
                "__metadata__": {
                    "ss_base_model_version": "wan2.2_14b_t2v",
                    "ss_network_dim": "32",
                    "ss_unrelated_key": "ignored",
                }
            }
            header = json.dumps(header_obj).encode()
            f.write(struct.pack("<Q", len(header)))
            f.write(header)
            path = Path(f.name)
        meta = read_safetensors_metadata(path)
        self.assertEqual(meta.get("ss_base_model_version"), "wan2.2_14b_t2v")
        self.assertEqual(meta.get("ss_network_dim"), "32")
        self.assertNotIn("ss_unrelated_key", meta)


if __name__ == "__main__":
    unittest.main()
