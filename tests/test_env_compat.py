from fastretrieval.common.utils import define_cache_dir


def test_new_env_name_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("FASTRETRIEVAL_CACHE_PATH", str(tmp_path / "new"))
    monkeypatch.setenv("QWEN3_EMBED_CACHE_PATH", str(tmp_path / "old"))
    assert define_cache_dir() == tmp_path / "new"


def test_old_env_name_still_read(monkeypatch, tmp_path):
    monkeypatch.delenv("FASTRETRIEVAL_CACHE_PATH", raising=False)
    monkeypatch.setenv("QWEN3_EMBED_CACHE_PATH", str(tmp_path / "old"))
    assert define_cache_dir() == tmp_path / "old"


def test_old_env_name_warns(monkeypatch, tmp_path, recwarn):
    monkeypatch.delenv("FASTRETRIEVAL_CACHE_PATH", raising=False)
    monkeypatch.setenv("QWEN3_EMBED_CACHE_PATH", str(tmp_path / "old"))
    define_cache_dir()
    assert any("FASTRETRIEVAL_CACHE_PATH" in str(w.message) for w in recwarn)
