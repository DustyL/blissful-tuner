"""Tests for musubi_tuner.utils.cli_compat — shared CLI deprecation helpers.

Validates that:
1. add_lycoris_arg() registers both --prefer_lycoris and --lycoris alias
2. validate_lycoris_arg() emits deprecation warning for --lycoris
3. validate_lycoris_arg() raises when lycoris is not installed
4. All 8 generation scripts register the --prefer_lycoris flag via cli_compat
"""

import argparse
import ast
import unittest
from pathlib import Path
from unittest.mock import patch

from musubi_tuner.utils.cli_compat import add_lycoris_arg, validate_lycoris_arg

# All generation scripts that must register --prefer_lycoris via cli_compat
GENERATION_SCRIPTS = [
    "wan_generate_video.py",
    "hv_generate_video.py",
    "hv_1_5_generate_video.py",
    "fpack_generate_video.py",
    "flux_kontext_generate_image.py",
    "flux_2_generate_image.py",
    "zimage_generate_image.py",
    "qwen_image_generate_image.py",
]

SRC_DIR = Path(__file__).resolve().parent.parent / "src" / "musubi_tuner"


class TestAddLycorisArg(unittest.TestCase):
    def test_registers_prefer_lycoris(self):
        parser = argparse.ArgumentParser()
        add_lycoris_arg(parser)
        args = parser.parse_args(["--prefer_lycoris"])
        self.assertTrue(args.prefer_lycoris)

    def test_registers_lycoris_alias(self):
        parser = argparse.ArgumentParser()
        add_lycoris_arg(parser)
        args = parser.parse_args(["--lycoris"])
        self.assertTrue(args.prefer_lycoris)

    def test_default_is_false(self):
        parser = argparse.ArgumentParser()
        add_lycoris_arg(parser)
        args = parser.parse_args([])
        self.assertFalse(args.prefer_lycoris)


class TestValidateLycorisArg(unittest.TestCase):
    def test_warns_on_lycoris_flag(self):
        args = argparse.Namespace(prefer_lycoris=True)
        with patch("musubi_tuner.utils.cli_compat.logger") as mock_logger:
            with patch("musubi_tuner.utils.cli_compat.lycoris_available", True):
                validate_lycoris_arg(args, argv=["script", "--lycoris"])
            mock_logger.warning.assert_called_once()
            msg = mock_logger.warning.call_args[0][0]
            self.assertIn("deprecated", msg)
            self.assertIn("v0.14.0", msg)

    def test_no_warning_on_prefer_lycoris(self):
        args = argparse.Namespace(prefer_lycoris=True)
        with patch("musubi_tuner.utils.cli_compat.logger") as mock_logger:
            with patch("musubi_tuner.utils.cli_compat.lycoris_available", True):
                validate_lycoris_arg(args, argv=["script", "--prefer_lycoris"])
            mock_logger.warning.assert_not_called()

    def test_raises_when_lycoris_missing(self):
        args = argparse.Namespace(prefer_lycoris=True)
        with patch("musubi_tuner.utils.cli_compat.lycoris_available", False):
            with self.assertRaises(ValueError) as ctx:
                validate_lycoris_arg(args, argv=["script", "--prefer_lycoris"])
            self.assertIn("lycoris", str(ctx.exception).lower())

    def test_no_error_when_flag_not_set(self):
        args = argparse.Namespace(prefer_lycoris=False)
        with patch("musubi_tuner.utils.cli_compat.lycoris_available", False):
            validate_lycoris_arg(args, argv=["script"])  # should not raise

    def test_handles_missing_prefer_lycoris_attr(self):
        args = argparse.Namespace()  # no prefer_lycoris at all
        validate_lycoris_arg(args, argv=["script"])  # should not raise


class TestGenerationScriptsUseCLICompat(unittest.TestCase):
    """AST-level check that all generation scripts import and call cli_compat helpers."""

    def _get_ast(self, script_name):
        path = SRC_DIR / script_name
        self.assertTrue(path.exists(), f"{script_name} not found at {path}")
        return ast.parse(path.read_text())

    def _imports_name(self, tree, name):
        """Check if the AST imports `name` from cli_compat."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module and "cli_compat" in node.module:
                for alias in node.names:
                    if alias.name == name:
                        return True
        return False

    def _calls_function(self, tree, name):
        """Check if the AST contains a call to function `name`."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                func = node.func
                if isinstance(func, ast.Name) and func.id == name:
                    return True
        return False

    def test_all_scripts_import_add_lycoris_arg(self):
        for script in GENERATION_SCRIPTS:
            with self.subTest(script=script):
                tree = self._get_ast(script)
                self.assertTrue(
                    self._imports_name(tree, "add_lycoris_arg"),
                    f"{script} does not import add_lycoris_arg from cli_compat",
                )

    def test_all_scripts_import_validate_lycoris_arg(self):
        for script in GENERATION_SCRIPTS:
            with self.subTest(script=script):
                tree = self._get_ast(script)
                self.assertTrue(
                    self._imports_name(tree, "validate_lycoris_arg"),
                    f"{script} does not import validate_lycoris_arg from cli_compat",
                )

    def test_all_scripts_call_add_lycoris_arg(self):
        for script in GENERATION_SCRIPTS:
            with self.subTest(script=script):
                tree = self._get_ast(script)
                self.assertTrue(
                    self._calls_function(tree, "add_lycoris_arg"),
                    f"{script} does not call add_lycoris_arg(parser)",
                )

    def test_all_scripts_call_validate_lycoris_arg(self):
        for script in GENERATION_SCRIPTS:
            with self.subTest(script=script):
                tree = self._get_ast(script)
                self.assertTrue(
                    self._calls_function(tree, "validate_lycoris_arg"),
                    f"{script} does not call validate_lycoris_arg(args)",
                )

    def test_no_inline_lycoris_available(self):
        """Ensure scripts don't bypass cli_compat with their own find_spec('lycoris')."""
        for script in GENERATION_SCRIPTS:
            with self.subTest(script=script):
                source = (SRC_DIR / script).read_text()
                # Allow import of lycoris_available FROM cli_compat, but not inline definition
                self.assertNotIn(
                    'find_spec("lycoris")',
                    source,
                    f"{script} has inline find_spec('lycoris') — should use cli_compat.lycoris_available",
                )
                self.assertNotIn(
                    "find_spec('lycoris')",
                    source,
                    f"{script} has inline find_spec('lycoris') — should use cli_compat.lycoris_available",
                )


if __name__ == "__main__":
    unittest.main()
