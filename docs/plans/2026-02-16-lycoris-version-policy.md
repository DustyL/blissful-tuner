# Simplify LyCORIS Adapter Fallback: Define Minimum Version Policy

> **Status:** Deferred — needs policy decision
> **Tracks:** Post-LoKr cleanup PR 4

## Summary

The LyCORIS integration in `lora_utils.py` contains a fallback reconstruction path (`_reconstruct_lokr_weight`) that manually rebuilds LoKr weights when the LyCORIS library's own merge API is unavailable or incompatible. This adds maintenance burden and duplicates upstream logic.

**Blocked on:** Defining a minimum supported `lycoris-lora` version and deciding behavior when the version is below minimum.

## Background

- `merge_nonlora_to_model()` in `src/musubi_tuner/utils/lora_utils.py` attempts to use `lycoris.utils.merge.merge_loha()` / `merge_lokr()` when available
- If LyCORIS is not installed or the API doesn't exist, it falls back to a manual reconstruction path
- The manual path works but is fragile: it must track any upstream changes to LoKr/LoHa weight decomposition

## Policy Decisions Needed

### 1. Minimum supported `lycoris-lora` version

- **Options:**
  - `>= 2.3.0` (first version with stable `lycoris.utils.merge` API)
  - `>= 3.0.0` (if targeting latest API surface)
  - No minimum (keep fallback indefinitely)
- **Recommendation:** `>= 2.3.0` — covers the merge API we actually use

### 2. Behavior when version is below minimum

- **Options:**
  - **Hard error:** `raise ImportError("lycoris-lora >= 2.3.0 required")`
  - **Warning + fallback:** Warn but use reconstruction path
  - **Silent fallback:** Current behavior
- **Recommendation:** Warning + fallback for one release, then hard error

### 3. Behavior when LyCORIS is not installed but LoHa/LoKr weights detected

- **Current:** Falls through to reconstruction path
- **Recommended:** Hard error with install instructions (already partially implemented via `format_unknown_network_type_error`)

## Entry Criteria

- [ ] Minimum version policy decided (owner sign-off)
- [ ] Behavior matrix documented (installed+new, installed+old, not installed)

## Exit Criteria (Acceptance Checklist)

- [ ] Version check added to `merge_nonlora_to_model()` or import path
- [ ] Fallback reconstruction path either:
  - Removed (if hard error on old versions), OR
  - Guarded behind explicit version check with deprecation warning
- [ ] `pyproject.toml` updated if minimum version becomes a hard dependency
- [ ] Test added: mock `lycoris.__version__` below minimum → expected behavior
- [ ] Test added: LyCORIS not installed + LoHa/LoKr weights → clean error message
- [ ] Full test suite passes
- [ ] `ruff check` and `ruff format --check` pass

## Proposed Rollout

| Phase | Version | Action |
|-------|---------|--------|
| 1 | v0.13.x | Add version check, warn if < minimum, keep fallback |
| 2 | v0.14.0 | Hard error if < minimum, remove fallback reconstruction |
