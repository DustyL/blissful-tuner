"""Early runtime filters that must run before torch / transformers imports.

Call ``apply_early_runtime_filters()`` at the very top of each training
entrypoint, **before** ``import torch``.  The function is idempotent — safe
to call from multiple modules.

The primary target today is the multi-line FBGEMM_GPU compatibility warning
emitted by ``fbgemm_gpu.__init__`` via ``logging.warning()`` on the *root*
logger during ``import torch``.  Because fbgemm_gpu uses the root logger
directly (not a named logger), we must add a filter to the root logger.
"""

import logging

_applied: bool = False


class _FBGEMMRootFilter(logging.Filter):
    """Drop FBGEMM_GPU compatibility warnings from the root logger.

    fbgemm_gpu emits a multi-line ``logging.warning()`` when its version
    does not match the PyTorch compatibility table.  The call uses the root
    logger (``logging.warning(...)``), so we filter at root.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return "FBGEMM_GPU" not in record.getMessage()


def apply_early_runtime_filters() -> None:
    """Install early warning filters.  Idempotent — repeated calls are no-ops."""
    global _applied
    if _applied:
        return
    _applied = True

    # Suppress FBGEMM_GPU compatibility spam on the root logger.
    logging.getLogger().addFilter(_FBGEMMRootFilter())
