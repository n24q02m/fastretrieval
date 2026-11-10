"""Tests for the server Session adapter over the fastretrieval runtime.

Deterministic and offline: ``registry.TextEmbedding`` / ``registry.TextCrossEncoder``
are patched at the seam, so no model artifact is ever downloaded. Shape
conversion, dims validation and score ordering run against the real Session code.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import numpy as np
import pytest

from fastretrieval.server import registry
from fastretrieval.server.registry import DIM_MAX, DIM_MIN, Session

MODEL = "n24q02m/Qwen3-Embedding-0.6B-ONNX"


class _FakeTextEmbedding:
    """Records construction and returns fixed-width unit vectors."""

    instances: list[_FakeTextEmbedding] = []

    def __init__(
        self, model_name: str, cache_dir: str | None = None, threads: int | None = None, **kw
    ) -> None:
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self.calls: list[tuple[list[str], dict]] = []
        _FakeTextEmbedding.instances.append(self)

    def embed(self, documents: Iterable[str], **kw) -> Iterable[np.ndarray]:
        docs = list(documents)
        self.calls.append((docs, dict(kw)))
        dim = kw.get("dim", DIM_MAX)
        return [np.full(dim, 1.0 / np.sqrt(dim), dtype=np.float32) for _ in docs]


class _FakeTextCrossEncoder:
    instances: list[_FakeTextCrossEncoder] = []

    def __init__(
        self, model_name: str, cache_dir: str | None = None, threads: int | None = None, **kw
    ) -> None:
        self.model_name = model_name
        _FakeTextCrossEncoder.instances.append(self)

    def rerank(self, query: str, documents: Iterable[str], **kw) -> Iterable[float]:
        # Deterministic scores: doc index order is deliberately not sorted.
        return [0.2, 0.9, 0.5]


class _MissingRuntime:
    """What the library's model class raises when its runtime is absent."""

    def __init__(self, *a, **kw) -> None:
        raise ImportError(
            "llama-cpp-python is required for GGUF models. Install with: pip install fastretrieval[gguf]"
        )


@pytest.fixture(autouse=True)
def _reset_fakes() -> Iterator[None]:
    _FakeTextEmbedding.instances.clear()
    _FakeTextCrossEncoder.instances.clear()
    registry.reset_sessions()
    yield
    registry.reset_sessions()


def test_session_init() -> None:
    session = Session("path/to/model", "embed")
    assert session.model_name == "path/to/model"
    assert session.kind == "embed"


def test_embed_lazy_loads_on_first_use(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "TextEmbedding", _FakeTextEmbedding)
    session = Session(MODEL, "embed", cache_dir="/tmp/models", threads=2)
    # Construction must not touch the library.
    assert _FakeTextEmbedding.instances == []

    out = session.embed(["hello", "world"], dims=256)

    assert len(out) == 2
    assert len(out[0]) == 256
    assert out[0][0] == pytest.approx(1.0 / np.sqrt(256))
    assert len(_FakeTextEmbedding.instances) == 1
    inst = _FakeTextEmbedding.instances[0]
    assert inst.model_name == MODEL
    assert inst.cache_dir == "/tmp/models"
    assert inst.threads == 2
    assert inst.calls == [(["hello", "world"], {"dim": 256})]


def test_embed_reuses_session(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "TextEmbedding", _FakeTextEmbedding)
    session = Session(MODEL, "embed")
    session.embed(["a"])
    session.embed(["b"])
    assert len(_FakeTextEmbedding.instances) == 1


def test_embed_rejects_out_of_range_dims() -> None:
    session = Session(MODEL, "embed")
    with pytest.raises(ValueError, match="dims"):
        session.embed(["x"], dims=DIM_MAX + 1)
    with pytest.raises(ValueError, match="dims"):
        session.embed(["x"], dims=DIM_MIN - 1)


def test_embed_returns_plain_float_lists(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "TextEmbedding", _FakeTextEmbedding)
    out = Session(MODEL, "embed").embed(["x"], dims=64)
    assert type(out) is list
    assert type(out[0]) is list
    assert all(type(v) is float for v in out[0])


def test_rerank_returns_index_score_pairs_sorted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "TextCrossEncoder", _FakeTextCrossEncoder)
    session = Session("n24q02m/Qwen3-Reranker-0.6B-ONNX", "rerank")
    ranked = session.rerank("q", ["d0", "d1", "d2"])
    assert ranked == [(1, 0.9), (2, 0.5), (0, 0.2)]
    # Rerank must not have built an embedding session.
    assert _FakeTextEmbedding.instances == []


def test_rerank_top_n(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "TextCrossEncoder", _FakeTextCrossEncoder)
    session = Session("n24q02m/Qwen3-Reranker-0.6B-ONNX", "rerank")
    assert session.rerank("q", ["d0", "d1", "d2"], top_n=2) == [(1, 0.9), (2, 0.5)]


def test_missing_gguf_runtime_propagates_library_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(registry, "TextEmbedding", _MissingRuntime)
    session = Session("n24q02m/Qwen3-Embedding-0.6B-GGUF", "embed")
    with pytest.raises(ImportError, match=r"fastretrieval\[gguf\]"):
        session.embed(["x"])


def test_missing_gguf_runtime_propagates_library_hint_for_rerank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry, "TextCrossEncoder", _MissingRuntime)
    session = Session("n24q02m/Qwen3-Reranker-0.6B-GGUF", "rerank")
    with pytest.raises(ImportError, match=r"fastretrieval\[gguf\]"):
        session.rerank("q", ["d"])
