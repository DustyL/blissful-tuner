# Deprecation Sunset: Remove --lycoris, --compile_args, --fp8_te Aliases

> **Status:** Deferred — waiting for release boundary
> **Tracks:** Post-LoKr cleanup PR 3

## Summary

Remove deprecated CLI argument aliases that have been superseded by their canonical replacements. These aliases currently emit deprecation warnings but remain functional.

**Blocked on:** A release boundary where users have had adequate warning.

## Scope

### 1. `--lycoris` → `--prefer_lycoris` (8 generation scripts)

- **Current state:** Centralized in `src/musubi_tuner/utils/cli_compat.py` via `add_lycoris_arg()` / `validate_lycoris_arg()`
- **Action:** Remove `"--lycoris"` from `add_lycoris_arg()`, remove `--lycoris` argv check from `validate_lycoris_arg()`, update help text
- **Files:** `src/musubi_tuner/utils/cli_compat.py`, `tests/test_cli_compat.py`

### 2. `--compile_args` → individual compile flags

- **Current state:** Registered in `blissful_core.py:442`, `wan_train_network.py:790`, `hv_generate_video.py:518`, `wan_generate_video.py:251`
- **Action:** Remove `--compile_args` argument and associated tuple-parsing shim
- **Docs:** `docs/torch_compile.md:219` notes the deprecation — update to remove mention

### 3. `--fp8_te` → `--fp8_text_encoder` (FLUX.2 only)

- **Current state:** Hidden alias in `flux_2_train_network.py:446`, warning at line 462
- **Action:** Remove `--fp8_te` argument and deprecation check

## Entry Criteria

- [ ] Version-pinned deprecation warnings are live for at least one release cycle
  - Proposed: Update warnings now to say "will be removed in v0.14.0"
  - Actual removal happens in v0.14.0
- [ ] CHANGELOG or release notes document the upcoming removal

## Exit Criteria (Acceptance Checklist)

- [ ] `--lycoris` flag removed from argparse registration
- [ ] `--compile_args` flag and tuple-parsing shim removed
- [ ] `--fp8_te` flag removed
- [ ] No `argparse.SUPPRESS`'d aliases remain for these flags
- [ ] `docs/torch_compile.md` updated (remove `--compile_args` deprecation note)
- [ ] `tests/test_cli_compat.py` updated (remove alias tests, add test that old flags raise clean errors)
- [ ] Full test suite passes
- [ ] `ruff check` and `ruff format --check` pass

## Proposed Timeline

| Milestone | Version | Action |
|-----------|---------|--------|
| Now | v0.12.x | Add version-pinned warning text ("removed in v0.14.0") |
| Next minor | v0.13.x | Warnings continue, users migrate |
| Removal | v0.14.0 | Delete aliases, update docs/tests |
