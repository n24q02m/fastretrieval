"""convert_gguf with a local model directory passed in the platform's own path form."""

import importlib.machinery
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from fastretrieval.convert import gguf as gguf_module
from fastretrieval.convert.gguf import convert_gguf

VOCAB = {"[PAD]": 0, "[UNK]": 1, "hello": 2, "world": 3}


def _write_model_dir(path: Path) -> Path:
    """Write a Hugging Face style folder with a real word-level tokenizer."""
    tokenizer = Tokenizer(models.WordLevel(VOCAB, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    config = {
        "architectures": ["Qwen3ForCausalLM"],
        "model_type": "qwen3",
        "hidden_size": 4,
        "max_position_embeddings": 16,
        "vocab_size": len(VOCAB),
    }
    path.mkdir(parents=True)
    (path / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (path / "tokenizer.json").write_text(tokenizer.to_str(), encoding="utf-8")
    (path / "tokenizer_config.json").write_text(
        json.dumps({"pad_token": "[PAD]", "unk_token": "[UNK]"}), encoding="utf-8"
    )
    return path


def _make_llama_checkout(root: Path) -> Path:
    (root / "build" / "bin").mkdir(parents=True)
    (root / "convert_hf_to_gguf.py").write_text("# converter entry point\n", encoding="utf-8")
    (root / "build" / "bin" / "llama-quantize").write_text("quantizer\n", encoding="utf-8")
    return root


class _FakeLlamaCpp:
    """Plays both llama.cpp subprocess steps by writing the files they would write."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def run(self, cmd: list[str], **options: object) -> subprocess.CompletedProcess[str]:
        self.calls.append(list(cmd))
        if Path(cmd[1]).name == "convert_hf_to_gguf.py":
            Path(cmd[cmd.index("--outfile") + 1]).write_bytes(b"GGUF f16\n")
        else:
            Path(cmd[2]).write_bytes(Path(cmd[1]).read_bytes() + cmd[3].encode("ascii"))
        return subprocess.CompletedProcess(cmd, 0, "", "")


@pytest.fixture
def llama(monkeypatch: pytest.MonkeyPatch) -> _FakeLlamaCpp:
    fake = _FakeLlamaCpp()
    # Boundary: convert_hf_to_gguf.py and llama-quantize are native llama.cpp subprocesses.
    monkeypatch.setattr(gguf_module.subprocess, "run", fake.run)
    # Boundary: the gguf package is only probed for presence by convert_gguf.
    module = types.ModuleType("gguf")
    module.__spec__ = importlib.machinery.ModuleSpec("gguf", None)
    monkeypatch.setitem(sys.modules, "gguf", module)
    return fake


@pytest.mark.parametrize("trailing", ["", os.sep], ids=["plain", "trailing-separator"])
def test_local_source_in_native_path_form_names_the_artifact_after_its_folder(
    tmp_path: Path, llama: _FakeLlamaCpp, trailing: str
) -> None:
    models_dir = tmp_path / "models"
    source = _write_model_dir(models_dir / "Tiny-Qwen3")
    llama_root = _make_llama_checkout(tmp_path / "llama.cpp")
    out = tmp_path / "artifacts" / "tiny-qwen3-gguf"

    # str(Path) uses the platform separator: a backslash on Windows.
    result = convert_gguf(
        str(source) + trailing,
        out,
        llama_cpp=str(llama_root),
        pooling="last_token",
        normalization=True,
    )

    assert result == out / "tiny-qwen3-q4-k-m.gguf"
    assert result.read_bytes() == b"GGUF f16\nQ4_K_M"
    # Both intermediate files stay inside the staging directory next to the output.
    (_, _, staged_model, _, f16, *_), (_, _, final, _) = llama.calls
    staging = Path(staged_model).parent
    assert staging.parent == out.parent
    assert Path(f16) == staging / "tiny-qwen3-f16.gguf"
    assert Path(final) == staging / "artifact" / "tiny-qwen3-q4-k-m.gguf"
    # Nothing is written next to the source model.
    assert sorted(path.name for path in models_dir.iterdir()) == ["Tiny-Qwen3"]
