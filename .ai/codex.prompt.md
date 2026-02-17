## About This File

This file provides guidance to Codex CLI when working with code in this repository.

## Guidelines

### Quick Orientation (Repo Conventions)
- Root-level `*.py` scripts are intentionally thin wrappers; the real entrypoints live in `src/musubi_tuner/` (and Blissful extensions in `src/blissful_tuner/`).
- When debugging behavior, prefer editing under `src/` and keep wrapper scripts minimal.

### Coding Style & Tooling
- Python: 4-space indentation. Ruff is configured in `pyproject.toml` (line length 132, formatter enabled).
- Naming: snake_case for files/functions (`*_train_network.py`, `*_generate_*`), PascalCase for classes.
- Types/Docs: Prefer type hints for public APIs and short docstrings describing args/returns.
- Lint: `ruff check` (optional auto-fix: `ruff check --fix`)
- Format: `ruff format src`
- Avoid broad refactors/formatting in vendored or excluded paths (see `tool.ruff.extend-exclude` / `tool.ruff.lint.per-file-ignores` in `pyproject.toml`).

### Testing Guidelines
- There is a `pytest` suite under `tests/`.
- Run: `pytest -q` (or `python -m pytest -q`).
- Prefer small, deterministic unit tests around data utilities, dataset/config parsing, cache formats, and argument parsing.

### Environment / Dependencies
- Primary source of truth is `pyproject.toml` (+ `uv.lock` when using `uv`).
- If a repo-local virtualenv exists at `./venv`, prefer running tools via `./venv/bin/python` to avoid accidentally using system Python.

### Commit & Pull Request Guidelines
- Commits: Use Conventional Commit style seen in history (`feat:`, `fix:`, `doc:`). Write clear, scoped messages.
- PRs: Include a summary, rationale, linked issue(s), and reproduction commands (e.g., the exact `python ... --args`). Add screenshots/log snippets when relevant.
- Docs: Update related files in `docs/` when changing behavior or flags.

### Security & Configuration Tips
- Large files: Do not commit datasets, model weights, or logs (`logs/` is ignored). Use external storage.
- Credentials: Keep any tokens/keys out of the repo and environment‑specific.
- CUDA: Choose the matching extra (`cu124`, `cu128`, `cu129`, `cu130`) for your driver; verify with `torch.cuda.is_available()`.
