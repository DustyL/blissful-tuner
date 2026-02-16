# Simplify LyCORIS Adapter Fallback: Define Minimum Version Policy

> **Status:** Deferred — policy decided, waiting for implementation slot
> **Tracks:** Post-LoKr cleanup PR 4

## Summary

The LyCORIS integration in `lora_utils.py` contains a fallback reconstruction path (`_reconstruct_lokr_weight`) that manually rebuilds LoKr weights when the LyCORIS library's own merge API is unavailable or incompatible. This adds maintenance burden and duplicates upstream logic.

## Background

- `merge_nonlora_to_model()` in `src/musubi_tuner/utils/lora_utils.py` attempts to use `lycoris.utils.merge.merge_loha()` / `merge_lokr()` when available
- If LyCORIS is not installed or the API doesn't exist, it falls back to a manual reconstruction path
- The manual path works but is fragile: it must track any upstream changes to LoKr/LoHa weight decomposition

## Policy Decisions (Decided)

### 1. Minimum supported `lycoris-lora` version: `>= 3.4.0`

Rationale: PR 4 depends on `lycoris.kohya.create_network_from_weights` being reliably present. Version 3.4.0 is the threshold where this helper path is stable.

**While fallback still exists:** broader compatibility is fine (no version floor).
**Once PR 4 lands:** enforce `>= 3.4.0` (or capability check + hard error if missing).

### 2. Behavior when version is below minimum

- **Phase 1 (v0.13.x):** Warning + fallback. Emit deprecation warning but continue using reconstruction path.
- **Phase 2 (v0.14.0):** Hard error. `raise ImportError("lycoris-lora >= 3.4.0 required for LoHa/LoKr support")`

### 3. Behavior when LyCORIS is not installed but LoHa/LoKr weights detected

- **Current:** Falls through to reconstruction path
- **Target:** Hard error with install instructions (already partially implemented via `format_unknown_network_type_error`)

## Entry Criteria

- [x] Minimum version policy decided (owner sign-off: >= 3.4.0)
- [ ] Behavior matrix documented (installed+new, installed+old, not installed)

## Exit Criteria (Acceptance Checklist)

- [ ] Version check added to `merge_nonlora_to_model()` or import path
- [ ] Fallback reconstruction path either:
  - Removed (if hard error on old versions), OR
  - Guarded behind explicit version check with deprecation warning
- [ ] `pyproject.toml` updated if minimum version becomes a hard dependency
- [ ] Test added: mock `lycoris.__version__` below 3.4.0 → expected behavior
- [ ] Test added: LyCORIS not installed + LoHa/LoKr weights → clean error message
- [ ] Full test suite passes
- [ ] `ruff check` and `ruff format --check` pass

## Proposed Rollout

| Phase | Version | Action |
|-------|---------|--------|
| 1 | v0.13.x | Add version check, warn if < 3.4.0, keep fallback |
| 2 | v0.14.0 | Hard error if < 3.4.0, remove fallback reconstruction |
