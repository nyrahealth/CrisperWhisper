"""Preload NVIDIA userspace libraries from pip-installed ``nvidia-*`` packages.

Release wheels of ``ctranslate2-crisperwhisper`` are built with
``CUDA_DYNAMIC_LOADING=ON``: instead of linking CUDA libraries, they
``dlopen("libcublas.so.12")`` by soname on first GPU use. That lookup only
searches the default loader paths, so a pip-installed
``nvidia-cublas-cu12`` (delivered by the ``crisperwhisper[ct2]`` extra) is
invisible without ``LD_LIBRARY_PATH`` gymnastics.

Loading the pip copies here with ``RTLD_GLOBAL`` *before* the dlopen happens
makes the soname lookup resolve to the already-loaded library — the same
trick PyTorch uses. No-op on non-Linux, when the nvidia packages are absent
(a system CUDA install then serves the dlopen), or when the libraries were
already loaded.
"""

from __future__ import annotations

import ctypes
import importlib.util
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# (module, soname globs in load order) — order matters: dependents last.
# Deliberately NOT libnvblas (wants NVBLAS_CONFIG_FILE, warns on stderr) —
# ctranslate2 only dlopens libcublas, which needs libcublasLt.
_PRELOAD_SPEC = [
    ("nvidia.cublas", ["libcublasLt.so.*", "libcublas.so.*"]),
]

_done = False


def preload() -> None:
    """Best-effort preload; never raises."""
    global _done
    if _done or sys.platform != "linux":
        _done = True
        return
    _done = True

    for module, patterns in _PRELOAD_SPEC:
        try:
            spec = importlib.util.find_spec(module)
        except (ImportError, ValueError):
            spec = None
        if spec is None or not spec.submodule_search_locations:
            continue
        lib_dir = Path(list(spec.submodule_search_locations)[0]) / "lib"
        if not lib_dir.is_dir():
            continue
        for pattern in patterns:
            for lib in sorted(lib_dir.glob(pattern)):
                try:
                    ctypes.CDLL(str(lib), mode=ctypes.RTLD_GLOBAL)
                    logger.debug("Preloaded %s", lib)
                except OSError as exc:
                    logger.debug("Could not preload %s: %s", lib, exc)
