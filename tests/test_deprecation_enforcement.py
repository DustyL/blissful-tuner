"""Deprecation enforcement tests for the v0.14.0 flag removal cycle.

Two test classes with complementary roles:

1. TestDeprecatedFlagsStillPresent (runs NOW)
   - Guards against premature removal of deprecated flags.
   - Ensures flags stay available until the coordinated v0.14.0 cleanup.
   - When v0.14.0 arrives, these tests should be DELETED along with the flags.

2. TestDeprecatedFlagsRemoved (SKIPPED until v0.14.0)
   - Enforces that the v0.14.0 removal is complete and clean.
   - Un-skips automatically when the version in pyproject.toml reaches 0.14.0.
   - Verifies old flags are gone from parsers and source, and replacements work.

See docs/DEPRECATION_NOTICES.md and docs/plans/2026-02-16-deprecation-sunset.md.
"""

import argparse
import ast
import re
import unittest
from pathlib import Path
from unittest.mock import patch

SRC_DIR = Path(__file__).resolve().parent.parent / "src" / "musubi_tuner"
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _get_project_version() -> tuple[int, int, int]:
    """Parse version from pyproject.toml as (major, minor, patch)."""
    text = (PROJECT_ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version\s*=\s*"(\d+)\.(\d+)\.(\d+)"', text, re.MULTILINE)
    assert match, "Could not parse version from pyproject.toml"
    return int(match.group(1)), int(match.group(2)), int(match.group(3))


def _source_has_argparse_flag(script_path: Path, flag_name: str) -> bool:
    """AST check: does the script register `flag_name` via parser.add_argument?"""
    tree = ast.parse(script_path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            # Match parser.add_argument("--flag_name", ...) or add_argument("--flag_name", ...)
            func = node.func
            is_add_arg = (isinstance(func, ast.Attribute) and func.attr == "add_argument") or (
                isinstance(func, ast.Name) and func.id == "add_argument"
            )
            if is_add_arg:
                for arg in node.args:
                    if isinstance(arg, ast.Constant) and arg.value == flag_name:
                        return True
    return False


PROJECT_VERSION = _get_project_version()
IS_V014_OR_LATER = PROJECT_VERSION >= (0, 14, 0)

# Scripts that register --compile_args
COMPILE_ARGS_SCRIPTS = [
    "hv_generate_video.py",
    "wan_generate_video.py",
]


# =============================================================================
# Pre-removal guards: ensure deprecated flags are NOT removed prematurely
# =============================================================================


@unittest.skipIf(IS_V014_OR_LATER, "Deprecated flags should be removed in v0.14.0 — delete this test class")
class TestDeprecatedFlagsStillPresent(unittest.TestCase):
    """Guard: deprecated flags must remain present until v0.14.0."""

    def test_lycoris_alias_still_registered(self):
        """--lycoris must remain as an alias in cli_compat until v0.14.0."""
        from musubi_tuner.utils.cli_compat import add_lycoris_arg

        parser = argparse.ArgumentParser()
        add_lycoris_arg(parser)
        # Should NOT raise
        args = parser.parse_args(["--lycoris"])
        self.assertTrue(args.prefer_lycoris)

    def test_compile_args_still_registered(self):
        for script in COMPILE_ARGS_SCRIPTS:
            with self.subTest(script=script):
                self.assertTrue(
                    _source_has_argparse_flag(SRC_DIR / script, "--compile_args"),
                    f"{script} should still register --compile_args until v0.14.0",
                )

    def test_fp8_te_still_registered(self):
        self.assertTrue(
            _source_has_argparse_flag(SRC_DIR / "flux_2_train_network.py", "--fp8_te"),
            "flux_2_train_network.py should still register --fp8_te until v0.14.0",
        )


# =============================================================================
# Post-removal enforcement: verify clean removal in v0.14.0+
# =============================================================================


@unittest.skipUnless(IS_V014_OR_LATER, "Removal enforcement activates in v0.14.0")
class TestDeprecatedFlagsRemoved(unittest.TestCase):
    """Enforce: deprecated flags must be fully removed in v0.14.0."""

    # --- --lycoris ---

    def test_lycoris_alias_removed_from_parser(self):
        """--lycoris must no longer be accepted by add_lycoris_arg."""
        from musubi_tuner.utils.cli_compat import add_lycoris_arg

        parser = argparse.ArgumentParser()
        add_lycoris_arg(parser)
        with self.assertRaises(SystemExit):
            parser.parse_args(["--lycoris"])

    def test_prefer_lycoris_still_works(self):
        """--prefer_lycoris must continue to work after alias removal."""
        from musubi_tuner.utils.cli_compat import add_lycoris_arg

        parser = argparse.ArgumentParser()
        add_lycoris_arg(parser)
        args = parser.parse_args(["--prefer_lycoris"])
        self.assertTrue(args.prefer_lycoris)

    def test_lycoris_argv_check_removed(self):
        """validate_lycoris_arg should not check for --lycoris in argv."""
        from musubi_tuner.utils.cli_compat import validate_lycoris_arg

        args = argparse.Namespace(prefer_lycoris=False)
        # Should not warn even if --lycoris appears in argv (flag is gone)
        with patch("musubi_tuner.utils.cli_compat.logger") as mock_logger:
            validate_lycoris_arg(args, argv=["script", "--lycoris"])
            mock_logger.warning.assert_not_called()

    # --- --compile_args ---

    def test_compile_args_removed_from_generators(self):
        """--compile_args must no longer be registered in generation scripts."""
        for script in COMPILE_ARGS_SCRIPTS:
            with self.subTest(script=script):
                self.assertFalse(
                    _source_has_argparse_flag(SRC_DIR / script, "--compile_args"),
                    f"{script} still registers --compile_args — remove it for v0.14.0",
                )

    def test_compile_args_shim_removed(self):
        """The tuple-unpacking shim for --compile_args must be removed."""
        for script in COMPILE_ARGS_SCRIPTS:
            with self.subTest(script=script):
                source = (SRC_DIR / script).read_text()
                self.assertNotIn(
                    "args.compile_args",
                    source,
                    f"{script} still references args.compile_args — remove the shim for v0.14.0",
                )

    # --- --fp8_te ---

    def test_fp8_te_removed_from_flux2_trainer(self):
        """--fp8_te must no longer be registered in flux_2_train_network.py."""
        self.assertFalse(
            _source_has_argparse_flag(SRC_DIR / "flux_2_train_network.py", "--fp8_te"),
            "flux_2_train_network.py still registers --fp8_te — remove it for v0.14.0",
        )

    def test_fp8_te_shim_removed(self):
        """The fp8_te → fp8_text_encoder shim must be removed."""
        source = (SRC_DIR / "flux_2_train_network.py").read_text()
        self.assertNotIn(
            "fp8_te",
            source,
            "flux_2_train_network.py still references fp8_te — remove the shim for v0.14.0",
        )

    def test_fp8_text_encoder_still_works(self):
        """--fp8_text_encoder must continue to work after alias removal."""
        self.assertTrue(
            _source_has_argparse_flag(SRC_DIR / "flux_2_train_network.py", "--fp8_text_encoder"),
            "flux_2_train_network.py must still register --fp8_text_encoder",
        )


if __name__ == "__main__":
    unittest.main()
