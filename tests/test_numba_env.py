import os


def test_ensure_numba_cache_dir_creates_writable_path(monkeypatch):
    monkeypatch.delenv("NUMBA_CACHE_DIR", raising=False)

    from harmonydagger.numba_env import ensure_numba_cache_dir

    path = ensure_numba_cache_dir()

    assert os.path.isdir(path)
    assert os.access(path, os.W_OK)
    assert os.environ["NUMBA_CACHE_DIR"] == path


def test_ensure_numba_cache_dir_keeps_existing_value(monkeypatch, tmp_path):
    monkeypatch.setenv("NUMBA_CACHE_DIR", str(tmp_path))

    from harmonydagger.numba_env import ensure_numba_cache_dir

    path = ensure_numba_cache_dir()

    assert path == str(tmp_path)
    assert os.environ["NUMBA_CACHE_DIR"] == str(tmp_path)
