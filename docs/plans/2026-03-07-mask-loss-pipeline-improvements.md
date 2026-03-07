# Mask Loss Pipeline Improvements

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Three improvements from code review: float32 mask processing precision, area-scale+prior validation warning, CLAUDE.md doc fix.

**Architecture:** All changes are isolated to `mask_loss.py` and `CLAUDE.md`. No changes to the training loop, cache format, or any architecture-specific code. The mask processing dtype change keeps masks in float32 through blur/gamma/min_weight, only casting to `loss.dtype` at the broadcast multiply points.

**Tech Stack:** Python 3.12, PyTorch, unittest, ruff

---

### Task 0: Create feature branch

**Step 1: Create and checkout branch**

Run: `git checkout -b fix/mask-loss-pipeline-improvements`

**Step 2: Verify clean state**

Run: `git status`
Expected: clean working tree on new branch

---

### Task 1: Float32 Mask Processing — Test

**Files:**
- Modify: `tests/test_mask_loss.py` (add new test after `test_bfloat16_loss_accepts_float16_mask_weights` at line 313)

**Step 1: Write the failing test**

Add the following test method to `TestMaskedLossWithPrior` after line 313:

```python
    def test_gamma_processing_uses_float32_precision(self) -> None:
        """Mask gamma/min_weight processing should use float32 for precision.

        bfloat16 has ~3.3 decimal digits. For mask=0.002 with gamma=0.3,
        the power operation in bf16 vs float32 produces measurably different results.
        This test verifies the result matches float32 reference, not bf16.
        """
        # Use a mask value and gamma that expose bf16 precision limits
        mask_val = 0.002
        gamma = 0.3
        loss = torch.ones(1, 16, 1, 4, 4, dtype=torch.bfloat16)
        mask_weights = torch.full((1, 1, 4, 4), mask_val, dtype=torch.float16)

        args = argparse.Namespace(
            use_mask_loss=True,
            mask_gamma=gamma,
            mask_min_weight=0.0,
            mask_blur_kernel_size=0,
            mask_area_scale_beta=0.0,
            prior_preservation_weight=0.0,
            prior_mask_threshold=None,
            normalize_per_sample=False,
        )

        out = apply_masked_loss_with_prior(
            loss,
            mask_weights,
            prior_loss_unreduced=None,
            args=args,
            layout="video",
        )

        # Reference: float32 precision result
        # With all-ones loss and uniform mask, weighted mean = 1.0 regardless of mask value
        # (sum(1*w) / sum(w) = sum(w)/sum(w) = 1.0). So we verify via stats instead.
        stats: dict[str, torch.Tensor] = {}
        _ = apply_masked_loss_with_prior(
            loss,
            mask_weights.clone(),
            prior_loss_unreduced=None,
            stats=stats,
            args=args,
            layout="video",
        )

        # The processed mask mean should match float32 gamma: 0.002^0.3
        f32_ref = float(mask_val**gamma)
        bf16_ref = float(torch.tensor(mask_val, dtype=torch.bfloat16) ** gamma)
        processed_mean = stats["mask/processed_mean"].item()

        # Should match float32 reference, not bf16
        self.assertAlmostEqual(processed_mean, f32_ref, places=4,
            msg=f"Mask processing should use float32 precision: got {processed_mean}, "
                f"float32 ref={f32_ref}, bf16 ref={bf16_ref}")
```

**Step 2: Run the test to verify it fails**

Run: `/Users/dustin/blissful-tuner/venv/bin/pytest tests/test_mask_loss.py::TestMaskedLossWithPrior::test_gamma_processing_uses_float32_precision -v --tb=short`

Expected: FAIL — the current code processes gamma in bf16 (loss.dtype), so `processed_mean` will match `bf16_ref`, not `f32_ref`.

---

### Task 2: Float32 Mask Processing — Implementation

**Files:**
- Modify: `src/musubi_tuner/modules/mask_loss.py:668-670,767,822`

**Step 1: Change mask dtype cast to float32**

In `apply_masked_loss_with_prior`, replace lines 668-670:

```python
    # Ensure mask weights match loss device/dtype to prevent mixed-precision collisions.
    # Note: mask_weights may be stored as float16 in cache files to reduce disk I/O.
    mask_weights = mask_weights.to(loss.device, dtype=loss.dtype)
```

With:

```python
    # Move mask to loss device but keep in float32 for precision during processing
    # (gamma, min_weight, blur). Cast to loss.dtype only at broadcast multiply points.
    # Note: mask_weights may be stored as float16 in cache files to reduce disk I/O.
    mask_weights = mask_weights.to(device=loss.device, dtype=torch.float32)
```

**Step 2: Cast mask_processed at target loss broadcast multiply**

Replace line 767:

```python
    target_loss_weighted = loss * mask_processed
```

With:

```python
    target_loss_weighted = loss * mask_processed.to(dtype=loss.dtype)
```

**Step 3: Cast prior_mask at prior loss broadcast multiply**

Replace line 822:

```python
            prior_loss_weighted = prior_loss_unreduced * prior_mask
```

With:

```python
            prior_loss_weighted = prior_loss_unreduced * prior_mask.to(dtype=loss.dtype)
```

**Step 4: Run the new test to verify it passes**

Run: `/Users/dustin/blissful-tuner/venv/bin/pytest tests/test_mask_loss.py::TestMaskedLossWithPrior::test_gamma_processing_uses_float32_precision -v --tb=short`

Expected: PASS

**Step 5: Run the full mask loss test suite for regressions**

Run: `/Users/dustin/blissful-tuner/venv/bin/pytest tests/test_mask_loss.py tests/test_wan_mask_loss_integration.py -v --tb=short`

Expected: All tests PASS (no regressions)

**Step 6: Commit**

```bash
git add src/musubi_tuner/modules/mask_loss.py tests/test_mask_loss.py
git commit -m "fix(mask-loss): use float32 precision for mask gamma/min_weight processing

Mask weights are now kept in float32 through blur, gamma, and min_weight
transformations, only casting to loss.dtype at the broadcast multiply
points. This prevents bf16/fp16 precision loss during power operations
(e.g., mask^0.3 on small values). The compact mask (B,1,F,H,W) is
negligible VRAM compared to the full loss tensor."
```

---

### Task 3: Area-Scale Beta + Prior Warning — Test

**Files:**
- Modify: `tests/test_mask_loss.py` (add new test to `TestValidateMaskLossArgs` after `test_raises_error_for_negative_mask_area_scale_beta` at line 778)

**Step 1: Write the failing test**

Add the following test method to `TestValidateMaskLossArgs` after line 778:

```python
    def test_warns_area_scale_beta_with_prior_preservation(self) -> None:
        args = argparse.Namespace(
            use_mask_loss=True,
            prior_preservation_weight=1.0,
            prior_mask_threshold=None,
            mask_gamma=1.0,
            mask_min_weight=0.0,
            mask_blur_kernel_size=0,
            mask_area_scale_beta=0.5,
            normalize_per_sample=False,
        )

        with self.assertLogs("musubi_tuner.modules.mask_loss", level="WARNING") as cm:
            validate_mask_loss_args(args)

        joined = "\n".join(cm.output)
        self.assertIn("mask_area_scale_beta", joined)
        self.assertIn("prior", joined.lower())
```

**Step 2: Run the test to verify it fails**

Run: `/Users/dustin/blissful-tuner/venv/bin/pytest tests/test_mask_loss.py::TestValidateMaskLossArgs::test_warns_area_scale_beta_with_prior_preservation -v --tb=short`

Expected: FAIL — no warning is currently emitted for this combination.

---

### Task 4: Area-Scale Beta + Prior Warning — Implementation

**Files:**
- Modify: `src/musubi_tuner/modules/mask_loss.py:317-318` (insert after area_scale_beta validation, before line 319)

**Step 1: Add the warning**

Insert between line 317 (`raise ValueError(...)`) and line 319 (`if prior_preservation_weight > 0 and mask_min_weight > 0:`):

```python
    if mask_area_scale_beta > 0 and prior_preservation_weight > 0:
        _logger.warning(
            f"--mask_area_scale_beta={mask_area_scale_beta} with --prior_preservation_weight={prior_preservation_weight}: "
            "Area-scale beta reduces target loss for small masks, but prior loss is independently normalized. "
            "For tiny masks (<10% coverage), training may become prior-dominated. "
            "Consider reducing --prior_preservation_weight or --mask_area_scale_beta if target learning is too weak."
        )
```

**Step 2: Run the new test to verify it passes**

Run: `/Users/dustin/blissful-tuner/venv/bin/pytest tests/test_mask_loss.py::TestValidateMaskLossArgs::test_warns_area_scale_beta_with_prior_preservation -v --tb=short`

Expected: PASS

**Step 3: Run full validation tests for regressions**

Run: `/Users/dustin/blissful-tuner/venv/bin/pytest tests/test_mask_loss.py::TestValidateMaskLossArgs -v --tb=short`

Expected: All tests PASS

**Step 4: Commit**

```bash
git add src/musubi_tuner/modules/mask_loss.py tests/test_mask_loss.py
git commit -m "fix(mask-loss): warn when area-scale beta and prior preservation are both active

When --mask_area_scale_beta > 0 and --prior_preservation_weight > 0,
tiny masks can make training prior-dominated because area-scale reduces
target loss but prior loss is independently normalized. Adds a
validation warning to help users diagnose weak target learning."
```

---

### Task 5: CLAUDE.md Documentation Fix

**Files:**
- Modify: `CLAUDE.md:384`

**Step 1: Fix the mask cache dtype reference**

On line 384, change:

```
mask_weights_{F}x{H}x{W}_float32
```

To:

```
mask_weights_{F}x{H}x{W}_float16
```

**Step 2: Verify no other stale float32 references for mask_weights**

Run: `grep -n "mask_weights.*float32" CLAUDE.md`

Expected: No matches (the only reference was line 384).

**Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "doc(mask-loss): fix mask cache dtype reference from float32 to float16

All architectures (WAN, FLUX.2, Qwen-Image, Z-Image, Kandinsky 5)
save mask_weights as float16 in cache files. The CLAUDE.md reference
was outdated."
```

---

### Task 6: Final Verification

**Step 1: Run ruff checks**

Run: `/Users/dustin/blissful-tuner/venv/bin/python -m ruff check src/musubi_tuner/modules/mask_loss.py tests/test_mask_loss.py && /Users/dustin/blissful-tuner/venv/bin/python -m ruff format --check src/musubi_tuner/modules/mask_loss.py tests/test_mask_loss.py`

Expected: All checks passed, files already formatted.

**Step 2: Run full test suite**

Run: `/Users/dustin/blissful-tuner/venv/bin/pytest tests/test_mask_loss.py tests/test_wan_mask_loss_integration.py tests/test_lora_ema_teacher.py tests/test_prior_scheduling.py tests/test_loss_utils.py tests/test_wan_mask_spatial_validation.py tests/test_mask_loss_disabled_warning.py -v --tb=short`

Expected: All 56 tests PASS (54 existing + 2 new).

**Step 3: Syntax check**

Run: `/Users/dustin/blissful-tuner/venv/bin/python -m py_compile src/musubi_tuner/modules/mask_loss.py`

Expected: No output (success).
