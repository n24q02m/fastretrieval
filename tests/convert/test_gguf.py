import subprocess
import sys

import pytest

from fastretrieval.convert import gguf as gguf_module
from fastretrieval.convert import onnx as onnx_module
from fastretrieval.convert.gguf import _run, convert_gguf, find_llama_cpp


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


def test_missing_checkout_without_env_explains_how_to_build(monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_HOME", raising=False)

    with pytest.raises(FileNotFoundError, match=r"checkout not found\. Set LLAMA_CPP_HOME"):
        find_llama_cpp()


def test_run_returns_quietly_when_the_step_succeeds():
    assert _run([sys.executable, "-c", "print('ok')"], "probe") is None


def test_run_reports_the_exit_code_of_a_failed_step():
    with pytest.raises(RuntimeError, match="^probe exited with code 3$"):
        _run([sys.executable, "-c", "import sys; sys.exit(3)"], "probe")


def test_run_turns_a_timeout_into_a_runtime_error(monkeypatch):
    calls = []

    # Boundary (b): llama.cpp steps run as subprocesses; waiting out the real
    # 3600 s timeout is impossible in a unit test, so the call raises it directly.
    def timed_out(cmd, **kwargs):
        calls.append(kwargs)
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(gguf_module.subprocess, "run", timed_out)

    with pytest.raises(RuntimeError, match="^llama-quantize timed out$") as excinfo:
        _run(["llama-quantize"], "llama-quantize")

    assert isinstance(excinfo.value.__cause__, subprocess.TimeoutExpired)
    assert calls == [{"capture_output": True, "text": True, "check": False, "timeout": 3600}]


# ---------------------------------------------------------------------------
# Artifact promotion: the GGUF and ONNX converters share the same contract.
# ---------------------------------------------------------------------------
PROMOTERS = pytest.mark.parametrize(
    "promote",
    [gguf_module._promote_artifact_directory, onnx_module._promote_artifact_directory],
    ids=["gguf", "onnx"],
)


def _staged(tmp_path):
    staging = tmp_path / "staging"
    (staging / "onnx").mkdir(parents=True)
    (staging / "onnx" / "model.onnx").write_bytes(b"model")
    return staging


@PROMOTERS
def test_promotion_moves_the_staged_artifact_to_a_new_output_path(tmp_path, promote):
    staging = _staged(tmp_path)
    out_dir = tmp_path / "out"

    promote(staging, out_dir)

    assert not staging.exists()
    assert (out_dir / "onnx" / "model.onnx").read_bytes() == b"model"


@PROMOTERS
def test_promotion_replaces_an_existing_empty_output_directory(tmp_path, promote):
    staging = _staged(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()

    promote(staging, out_dir)

    assert not staging.exists()
    assert (out_dir / "onnx" / "model.onnx").read_bytes() == b"model"


@PROMOTERS
def test_promotion_refuses_an_output_path_that_is_a_file(tmp_path, promote):
    staging = _staged(tmp_path)
    out_file = tmp_path / "out"
    out_file.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output path is not a directory"):
        promote(staging, out_file)

    assert out_file.read_text(encoding="utf-8") == "keep"
    assert (staging / "onnx" / "model.onnx").is_file()


@PROMOTERS
def test_promotion_refuses_to_overwrite_a_non_empty_output_directory(tmp_path, promote):
    staging = _staged(tmp_path)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    (out_dir / "previous.txt").write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output directory is not empty"):
        promote(staging, out_dir)

    assert [path.name for path in out_dir.iterdir()] == ["previous.txt"]
    assert (staging / "onnx" / "model.onnx").is_file()
