"""Shared stand-in for the ``onnxruntime`` module, plus real tokenizer fixtures.

onnxruntime publishes no CPython 3.15 wheel, so tests that drive a model through its real
load path replace the module object imported by ``fastretrieval.common.onnx_model`` with
:func:`install_stub_runtime`. Everything else stays real: tokenizers are genuine
``tokenizers.Tokenizer`` files written to a temporary model directory and loaded by the
library's own ``load_tokenizer``.
"""

import json
from collections.abc import Callable, Iterable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

import fastretrieval.common.onnx_model as onnx_model_module

PAD_TOKEN = "<pad>"
UNK_TOKEN = "[UNK]"


def write_tokenizer_files(model_dir: Path, words: Iterable[str]) -> dict[str, int]:
    """Write a whitespace word-level tokenizer and its configs; return the vocabulary."""
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for word in words:
        vocab.setdefault(word, len(vocab))
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    model_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(str(model_dir / "tokenizer.json"))
    (model_dir / "config.json").write_text(json.dumps({"pad_token_id": 0}), encoding="utf-8")
    (model_dir / "tokenizer_config.json").write_text(
        json.dumps({"model_max_length": 512, "pad_token": PAD_TOKEN}), encoding="utf-8"
    )
    return vocab


class StubSession:
    """Replaces ``onnxruntime.InferenceSession``; ``compute`` maps the feed to output 0."""

    def __init__(
        self,
        compute: Callable[[dict[str, np.ndarray]], np.ndarray],
        input_names: Iterable[str] = ("input_ids", "attention_mask"),
    ) -> None:
        self._compute = compute
        self._input_names = tuple(input_names)
        self.feeds: list[dict[str, np.ndarray]] = []

    def get_inputs(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(name=name, shape=["batch_size", "sequence_length"])
            for name in self._input_names
        ]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="output", shape=["batch_size", "sequence_length", "dim"])]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(self, output_names: Any, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.feeds.append(feed)
        return [self._compute(feed)]


def install_stub_runtime(monkeypatch: pytest.MonkeyPatch, session: StubSession) -> list[str]:
    """Route every ONNX session creation to ``session``; return the opened model paths."""
    opened: list[str] = []

    def open_session(path: str, providers: Any, sess_options: Any) -> StubSession:
        opened.append(path)
        return session

    runtime = SimpleNamespace(
        get_available_providers=lambda: ["CPUExecutionProvider"],
        SessionOptions=SimpleNamespace,
        GraphOptimizationLevel=SimpleNamespace(ORT_ENABLE_ALL=99),
        ExecutionMode=SimpleNamespace(ORT_PARALLEL=1),
        InferenceSession=open_session,
    )
    monkeypatch.setattr(onnx_model_module, "ort", runtime)
    return opened
