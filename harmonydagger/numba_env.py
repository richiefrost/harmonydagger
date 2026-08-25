"""Point Numba at a writable cache directory before Librosa JIT compiles."""
from __future__ import annotations

import os
import sys
import tempfile

_DEFAULT_CACHE_NAME = "harmonydagger-numba-cache"


def ensure_numba_cache_dir() -> str:
    """Ensure NUMBA_CACHE_DIR exists and is writable.

    Librosa JIT-compiles helpers with ``cache=True``. If Numba cannot
    write next to site-packages or in the user cache dir, it raises
    ``RuntimeError: cannot cache function ... no locator available``.
    """
    cache_dir = _writable_cache_dir()
    os.environ["NUMBA_CACHE_DIR"] = cache_dir
    numba = sys.modules.get("numba")
    if numba is not None:
        config = getattr(numba, "config", None)
        if config is not None:
            config.CACHE_DIR = cache_dir
    return cache_dir


def _writable_cache_dir() -> str:
    existing = os.environ.get("NUMBA_CACHE_DIR")
    candidates = []
    if existing:
        candidates.append(existing)
    candidates.append(os.path.join(tempfile.gettempdir(), _DEFAULT_CACHE_NAME))
    candidates.append(os.path.join(os.getcwd(), ".numba_cache"))
    for path in candidates:
        try:
            os.makedirs(path, exist_ok=True)
            tempfile.TemporaryFile(dir=path).close()
            return path
        except OSError:
            continue
    raise OSError("Could not create a writable Numba cache directory")
