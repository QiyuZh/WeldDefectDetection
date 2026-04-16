from __future__ import annotations

from .bootstrap import bootstrap_windows_runtime

bootstrap_windows_runtime()

__all__ = [
    "__version__",
]

__version__ = "0.1.0"
