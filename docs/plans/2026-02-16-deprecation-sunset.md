# Deprecation Sunset: Remove --lycoris, --compile_args, --fp8_te Aliases

> **Status:** In progress — warnings live, enforcement tests staged, awaiting v0.14.0 branch cut
> **Tracks:** Post-LoKr cleanup PR 3
> **Draft PR:** `deprecation/v0.14.0-removals`

## Summary

Remove deprecated CLI argument aliases that have been superseded by their canonical replacements. These aliases currently emit deprecation warnings but remain functional.

## Scope

### 1. `--lycoris` → `--prefer_lycoris` (8 generation scripts)

- **Current state:** Centralized in `src/musubi_tuner/utils/cli_compat.py` via `add_lycoris_arg()` / `validate_lycoris_arg()`
- **Action:** Remove `"--lycoris"` from `add_lycoris_arg()`, remove `--lycoris` argv check from `validate_lycoris_arg()`, update help text
- **Files:** `src/musubi_tuner/utils/cli_compat.py`, `tests/test_cli_compat.py`

### 2. `--compile_args` → individual compile flags

- **Current state:** Registered in `hv_generate_video.py`, `wan_generate_video.py` (user-facing, default=None); also in `blissful_core.py`, `wan_train_network.py` (internal plumbing, non-None default)
- **Action:** Remove `--compile_args` from user-facing parsers, remove tuple-unpacking shim, remove from internal plumbing
- **Docs:** `docs/torch_compile.md:219` notes the deprecation — update to remove mention

### 3. `--fp8_te` → `--fp8_text_encoder` (FLUX.2 only)

- **Current state:** Hidden alias in `flux_2_train_network.py:446`, warning at line 462
- **Action:** Remove `--fp8_te` argument and deprecation check

## Entry Criteria

- [x] Version-pinned deprecation warnings are live (commit `6cdb89e`)
- [x] Deprecation notices documented (`docs/DEPRECATION_NOTICES.md`)
- [x] Enforcement test set staged (`tests/test_deprecation_enforcement.py`)
  - 3 pre-removal guards (run now, catch premature removal)
  - 8 post-removal enforcement tests (skip until v0.14.0, then verify clean removal)

## Exit Criteria (Acceptance Checklist)

When implementing the v0.14.0 removal:

- [ ] `--lycoris` flag removed from `add_lycoris_arg()` in `cli_compat.py`
- [ ] `--lycoris` argv check removed from `validate_lycoris_arg()` in `cli_compat.py`
- [ ] `--compile_args` registration removed from `hv_generate_video.py` and `wan_generate_video.py`
- [ ] `--compile_args` tuple-unpacking shim removed from both scripts
- [ ] `--compile_args` registration removed from `blissful_core.py` and `wan_train_network.py`
- [ ] `--compile_args` reference removed from `wan/modules/model.py`
- [ ] `--fp8_te` registration removed from `flux_2_train_network.py`
- [ ] `--fp8_te` → `--fp8_text_encoder` shim removed from `flux_2_train_network.py`
- [ ] No `argparse.SUPPRESS`'d aliases remain for these flags
- [ ] `docs/torch_compile.md` updated (remove `--compile_args` deprecation note)
- [ ] `docs/DEPRECATION_NOTICES.md` updated (move items to "Removed" section)
- [ ] `tests/test_cli_compat.py` updated (remove `--lycoris` alias tests)
- [ ] `tests/test_deprecation_enforcement.py`: delete `TestDeprecatedFlagsStillPresent` class
- [ ] `TestDeprecatedFlagsRemoved` tests un-skip and pass
- [ ] Full test suite passes
- [ ] `ruff check` and `ruff format --check` pass

## Timeline

| Milestone | Version | Action | Status |
|-----------|---------|--------|--------|
| Warnings | v0.12.x | Version-pinned deprecation warnings live | Done |
| Migration | v0.13.x | Warnings continue, users migrate | Pending |
| Removal | v0.14.0 | Delete aliases, update docs/tests | Staged |
