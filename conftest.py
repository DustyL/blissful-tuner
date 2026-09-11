"""Repo-root pytest configuration.

Its one job is to make the repo root the FIRST entry on ``sys.path`` so that this repo's own
``tools`` package wins over the one the editable PyTorch install brings in.

Why this is needed: pytest's default ``prepend`` import mode inserts the *test file's* directory
(``tests/``) on ``sys.path``, never the repo root, while the editable torch install does put
``~/pytorch`` there — and ``~/pytorch/tools/__init__.py`` is a real package. So
``from tools import merge_loras_algebra`` resolved to PyTorch's ``tools`` and raised ImportError
during collection. Because pytest aborts the whole run on a collection error, two unrelated test
modules took the entire suite down with them:

    tests/test_merge_loras_algebra.py   ImportError: cannot import name 'merge_loras_algebra' from 'tools'
    tests/test_lora_eval_inventory.py   ModuleNotFoundError: No module named 'tools.lora_eval'

The companion half of the fix is ``tools/__init__.py`` — see that file for why *both* are required.

Verified safe: of the 53 top-level wrapper scripts this exposes as importable modules, ``tools`` is
the only name that already resolved to something outside this repo, and it is exactly the one we
mean to take over.
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent)

# Prepend rather than append, and force position 0 even if the path is already present further
# down: once both `tools` packages are regular packages, first-on-the-path is what decides.
if _REPO_ROOT in sys.path:
    sys.path.remove(_REPO_ROOT)
sys.path.insert(0, _REPO_ROOT)

# Defensive: if anything loaded earlier than this conftest (a pytest plugin, say) already imported
# `tools` from outside the repo, the sys.path change above cannot dislodge the cached module. Drop
# the stale entry so the next import re-resolves against the corrected path. Scoped deliberately:
# only a `tools` whose origin lies outside this repo is evicted, and nothing in a pytest session
# legitimately depends on PyTorch's build-time `tools`.
_stale = sys.modules.get("tools")
if _stale is not None:
    _origin = getattr(_stale, "__file__", None) or ""
    if not _origin.startswith(_REPO_ROOT):
        for _name in [n for n in sys.modules if n == "tools" or n.startswith("tools.")]:
            del sys.modules[_name]
