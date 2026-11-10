"""Tests for fastretrieval.server.api.

The backend object is swapped at ``registry.get_backend`` -- the seam where
model artifacts load -- so request/response handling, alias resolution and
error mapping all run against the real code paths offline.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from fastretrieval.server import registry
from fastretrieval.server.api import __version__, app


class FakeBackend:
    """Deterministic backend: identity-preserving shapes, fixed scores."""

    def __init__(self) -> None:
        self.embed_calls: list[tuple[list[str], int]] = []
        self.rerank_calls: list[tuple[str, list[str], int | None]] = []

    def embed(self, texts: list[str], dims: int) -> list[list[float]]:
        self.embed_calls.append((texts, dims))
        # Deterministic vectors: v[j] = (i+1)*(j+1) normalized-ish values.
        return [[(i + 1) * (j + 1) / 100.0 for j in range(dims)] for i in range(len(texts))]

    def rerank(self, query: str, docs: list[str], top_n: int | None) -> list[tuple[int, float]]:
        self.rerank_calls.append((query, docs, top_n))
        scores = [0.2, 0.9, 0.5][: len(docs)]
        return sorted(enumerate(scores), key=lambda p: p[1], reverse=True)[: top_n or len(docs)]


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch):
    fake = FakeBackend()
    monkeypatch.setattr(registry, "get_backend", lambda requested, kind: fake)
    yield TestClient(app), fake


def test_health_returns_ok(client) -> None:
    c, _ = client
    resp = c.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": __version__}


def test_models_lists_alias_table(client) -> None:
    c, _ = client
    resp = c.get("/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["embed"]["qwen3-0.6b"] == "n24q02m/Qwen3-Embedding-0.6B-ONNX"
    assert body["rerank"]["qwen3-rerank-0.6b"] == "n24q02m/Qwen3-Reranker-0.6B-ONNX"


def test_embed_returns_vectors(client) -> None:
    c, fake = client
    resp = c.post("/embed", json={"input": ["hello", "world"], "dims": 8})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "n24q02m/Qwen3-Embedding-0.6B-ONNX"
    assert body["dims"] == 8
    assert len(body["data"]) == 2
    assert body["data"][0] == [0.01 * (j + 1) for j in range(8)]
    assert fake.embed_calls == [(["hello", "world"], 8)]


def test_embed_defaults(client) -> None:
    c, fake = client
    resp = c.post("/embed", json={"input": ["x"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["dims"] == 1024
    assert fake.embed_calls == [(["x"], 1024)]


def test_embed_mrl_truncation_dims_512(client) -> None:
    """dims=512 < 1024 native width: MRL-truncated vectors, one per input."""
    c, fake = client
    resp = c.post("/embed", json={"input": ["alpha", "beta"], "dims": 512})
    assert resp.status_code == 200
    body = resp.json()
    assert body["dims"] == 512
    assert len(body["data"]) == 2
    assert all(len(vector) == 512 for vector in body["data"])
    assert fake.embed_calls == [(["alpha", "beta"], 512)]


def test_embed_resolves_canonical_model_name(client) -> None:
    c, _ = client
    resp = c.post("/embed", json={"model": "custom/model", "input": ["x"], "dims": 4})
    assert resp.status_code == 200
    assert resp.json()["model"] == "custom/model"


def test_rerank_returns_sorted_results(client) -> None:
    c, fake = client
    resp = c.post("/rerank", json={"query": "q", "documents": ["a", "b", "c"]})
    assert resp.status_code == 200
    body = resp.json()
    assert body["model"] == "n24q02m/Qwen3-Reranker-0.6B-ONNX"
    assert body["results"] == [
        {"index": 1, "relevance_score": 0.9},
        {"index": 2, "relevance_score": 0.5},
        {"index": 0, "relevance_score": 0.2},
    ]
    assert fake.rerank_calls == [("q", ["a", "b", "c"], None)]


def test_rerank_top_n(client) -> None:
    c, fake = client
    resp = c.post("/rerank", json={"query": "q", "documents": ["a", "b", "c"], "top_n": 1})
    assert resp.status_code == 200
    assert resp.json()["results"] == [{"index": 1, "relevance_score": 0.9}]
    assert fake.rerank_calls == [("q", ["a", "b", "c"], 1)]


def test_embed_validates_input_schema(client) -> None:
    c, _ = client
    resp = c.post("/embed", json={})
    assert resp.status_code == 422


def test_rerank_validates_input_schema(client) -> None:
    c, _ = client
    resp = c.post("/rerank", json={})
    assert resp.status_code == 422


def test_embed_validates_types(client) -> None:
    c, _ = client
    resp = c.post("/embed", json={"input": "not a list"})
    assert resp.status_code == 422


def test_rerank_validates_types(client) -> None:
    c, _ = client
    resp = c.post("/rerank", json={"query": "test", "documents": "not a list"})
    assert resp.status_code == 422


def test_rerank_top_n_must_be_positive(client) -> None:
    c, _ = client
    resp = c.post("/rerank", json={"query": "q", "documents": ["a"], "top_n": 0})
    assert resp.status_code == 422


def test_unknown_model_maps_to_400(client, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Bad:
        def embed(self, texts, dims):
            raise ValueError("Model nope is not supported in TextEmbedding")

    monkeypatch.setattr(registry, "get_backend", lambda r, k: _Bad())
    resp = client[0].post("/embed", json={"model": "nope", "input": ["x"]})
    assert resp.status_code == 400
    assert "not supported" in resp.json()["detail"]


def test_missing_gguf_runtime_maps_to_503_with_hint(
    client, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Bad:
        def rerank(self, query, docs, top_n):
            raise ImportError(
                "llama-cpp-python is required for GGUF models. Install with: pip install fastretrieval[gguf]"
            )

    monkeypatch.setattr(registry, "get_backend", lambda r, k: _Bad())
    resp = client[0].post(
        "/rerank",
        json={"model": "qwen3-rerank-0.6b-gguf", "query": "q", "documents": ["d"]},
    )
    assert resp.status_code == 503
    assert "fastretrieval[gguf]" in resp.json()["detail"]


def test_backend_failure_maps_to_503(client, monkeypatch: pytest.MonkeyPatch) -> None:
    class _Bad:
        def embed(self, texts, dims):
            raise RuntimeError("download failed")

    monkeypatch.setattr(registry, "get_backend", lambda r, k: _Bad())
    resp = client[0].post("/embed", json={"input": ["x"]})
    assert resp.status_code == 503
    assert "download failed" in resp.json()["detail"]
