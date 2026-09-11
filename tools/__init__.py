"""Marks ``tools/`` as a regular package. **Load-bearing — do not delete.**

Without this file ``tools/`` is only a *namespace* package, and the editable PyTorch install puts
``~/pytorch`` on ``sys.path`` with its own regular ``tools`` package (``~/pytorch/tools/__init__.py``).
Python's path finder returns the first *regular* package it finds and treats namespace directories as
mere "portions" to keep searching past — so PyTorch's ``tools`` won regardless of ``sys.path`` order,
and `from tools import merge_loras_algebra` raised ImportError at pytest collection time.

Both halves of the fix are required, and neither alone is sufficient:
  1. this file, so blissful's ``tools`` is a regular package and can compete at all;
  2. the repo root prepended to ``sys.path`` (see the root ``conftest.py``), because once *both*
     are regular packages the winner is simply whichever comes first on the path.

``tools/lora_eval/`` deliberately needs no ``__init__.py``: once ``tools`` resolves here, its
subpackages are searched only within ``tools.__path__``, so there is nothing left to shadow them.

Intentionally empty otherwise — the modules in here are standalone CLI scripts, and importing them
eagerly would pull torch/PIL into every `import tools`.
"""
