from __future__ import annotations

import os
import sys
from pathlib import Path

_DLL_HANDLES: list[object] = []
_BOOTSTRAPPED = False


def bootstrap_windows_runtime() -> None:
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED or os.name != "nt":
        return

    prefix = Path(sys.prefix)
    candidates = [
        prefix,
        prefix / "Library" / "bin",
        prefix / "Library" / "usr" / "bin",
        prefix / "Library" / "mingw-w64" / "bin",
        prefix / "Scripts",
    ]

    current_path = os.environ.get("PATH", "")
    prepend_paths: list[str] = []
    for directory in candidates:
        if not directory.is_dir():
            continue
        directory_str = str(directory)
        if directory_str not in current_path:
            prepend_paths.append(directory_str)
        if hasattr(os, "add_dll_directory"):
            try:
                _DLL_HANDLES.append(os.add_dll_directory(directory_str))
            except (FileNotFoundError, OSError):
                continue

    if prepend_paths:
        os.environ["PATH"] = os.pathsep.join(prepend_paths + [current_path]) if current_path else os.pathsep.join(prepend_paths)

    _BOOTSTRAPPED = True

