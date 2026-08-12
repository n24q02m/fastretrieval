import pytest

from fastretrieval.convert.gguf import convert_gguf, find_llama_cpp


def _fake_checkout(tmp_path):
    root = tmp_path / "llama.cpp"
    (root / "build" / "bin").mkdir(parents=True)
    (root / "convert_hf_to_gguf.py").write_text("", encoding="utf-8")
    (root / "build" / "bin" / "llama-quantize").write_text("", encoding="utf-8")
    return root


def test_finds_checkout_from_env(tmp_path, monkeypatch):
    root = _fake_checkout(tmp_path)
    monkeypatch.setenv("LLAMA_CPP_HOME", str(root))
    assert find_llama_cpp() == root


def test_explicit_path_wins_over_env(tmp_path, monkeypatch):
    root = _fake_checkout(tmp_path)
    monkeypatch.setenv("LLAMA_CPP_HOME", str(tmp_path / "nope"))
    assert find_llama_cpp(str(root)) == root


def test_incomplete_checkout_names_the_missing_piece(tmp_path, monkeypatch):
    root = tmp_path / "llama.cpp"
    (root / "build" / "bin").mkdir(parents=True)
    (root / "convert_hf_to_gguf.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("LLAMA_CPP_HOME", str(root))
    with pytest.raises(FileNotFoundError, match="llama-quantize"):
        find_llama_cpp()


def test_missing_checkout_explains_how_to_build(tmp_path, monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_HOME", raising=False)
    with pytest.raises(FileNotFoundError, match="LLAMA_CPP_HOME"):
        find_llama_cpp(str(tmp_path / "absent"))


@pytest.mark.parametrize("bad", ["Q4_K_M; rm -rf /", "../../etc", "Q4 K M"])
def test_quant_type_is_validated(tmp_path, monkeypatch, bad):
    root = _fake_checkout(tmp_path)
    monkeypatch.setenv("LLAMA_CPP_HOME", str(root))
    with pytest.raises(ValueError, match="quantization type"):
        convert_gguf("acme/tiny", tmp_path / "out", quant=bad)
