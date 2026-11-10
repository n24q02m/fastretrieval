"""Model-name resolution and session cache for the HTTP server.

Public request model names (short aliases) map onto the names this package
registers; a name that matches no alias is passed through unchanged so an
operator can point at a custom or local model. ``TextEmbedding`` and
``TextCrossEncoder`` dispatch on the model name themselves (ONNX, GGUF or a
registered custom model), so a session holds exactly one of them and the
choice never appears here.

Sessions are cached per (kind, resolved name) so one server serves every
client off a single model instance. The cache directory is the library's own
(``FASTRETRIEVAL_CACHE_PATH``, resolved by ``define_cache_dir``); inference
threads come from ``FASTRETRIEVAL_THREADS``.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from fastretrieval.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder
from fastretrieval.text.text_embedding import TextEmbedding

# Qwen3-0.6B native width; MRL truncation accepts anything in [DIM_MIN, DIM_MAX].
DIM_MIN = 32
DIM_MAX = 1024

# Public alias -> registered model name. Aliases resolve case-insensitively.
EMBED_MODELS: dict[str, str] = {
    "qwen3-0.6b": "n24q02m/Qwen3-Embedding-0.6B-ONNX",
    "qwen3-0.6b-q4f16": "n24q02m/Qwen3-Embedding-0.6B-ONNX-Q4F16",
    "qwen3-0.6b-gguf": "n24q02m/Qwen3-Embedding-0.6B-GGUF",
}

RERANK_MODELS: dict[str, str] = {
    "qwen3-rerank-0.6b": "n24q02m/Qwen3-Reranker-0.6B-ONNX",
    "qwen3-rerank-0.6b-q4f16": "n24q02m/Qwen3-Reranker-0.6B-ONNX-Q4F16",
    "qwen3-rerank-0.6b-yesno": "n24q02m/Qwen3-Reranker-0.6B-ONNX-YesNo",
    "qwen3-rerank-0.6b-gguf": "n24q02m/Qwen3-Reranker-0.6B-GGUF",
}

THREADS_ENV = "FASTRETRIEVAL_THREADS"


def check_dims(dims: int, native: int = DIM_MAX) -> int:
    """Validate the requested output dimensionality against MRL bounds."""
    if dims < DIM_MIN or dims > native:
        raise ValueError(f"dims must be between {DIM_MIN} and {native} for this model, got {dims}")
    return dims


def as_float_matrix(embeddings: Any) -> list[list[float]]:
    """Convert an iterable of numpy vectors to a plain nested float list."""
    return [[float(v) for v in row] for row in embeddings]


def ranked_indices(scores: list[float], top_n: int | None = None) -> list[tuple[int, float]]:
    """Return ``(index, score)`` pairs sorted by score descending.

    Ties keep the original document order (stable sort on the index). ``top_n``
    caps the list; ``None`` returns every document.
    """
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    if top_n is not None:
        order = order[:top_n]
    return [(i, scores[i]) for i in order]


def resolve_model(requested: str, kind: str) -> str:
    """Map a request ``model`` name onto a registered model name.

    ``kind`` is ``"embed"`` or ``"rerank"`` and selects the alias table;
    unrecognised names pass through so the library can validate them.
    """
    table = EMBED_MODELS if kind == "embed" else RERANK_MODELS
    return table.get(requested.lower(), requested)


def _threads() -> int | None:
    raw = os.environ.get(THREADS_ENV)
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


class Session:
    """Lazy model session for one resolved name.

    ``model_name`` is a registered name (e.g.
    ``n24q02m/Qwen3-Embedding-0.6B-ONNX``) or an absolute local model
    directory. Nothing is downloaded or loaded until ``embed`` / ``rerank``
    is called; the first call resolves the model into the cache directory.
    """

    def __init__(
        self,
        model_name: str,
        kind: str,
        cache_dir: str | None = None,
        threads: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.kind = kind
        self.cache_dir = cache_dir
        self.threads = threads
        self._model: Any | None = None

    def _model_instance(self) -> Any:
        if self._model is None:
            kwargs: dict[str, Any] = {
                "model_name": self.model_name,
                "cache_dir": self.cache_dir,
                "threads": self.threads,
            }
            self._model = (
                TextEmbedding(**kwargs) if self.kind == "embed" else TextCrossEncoder(**kwargs)
            )
        return self._model

    def embed(self, texts: list[str], dims: int = DIM_MAX) -> list[list[float]]:
        check_dims(dims)
        return as_float_matrix(self._model_instance().embed(texts, dim=dims))

    def rerank(
        self, query: str, docs: list[str], top_n: int | None = None
    ) -> list[tuple[int, float]]:
        scores = [float(s) for s in self._model_instance().rerank(query, docs)]
        return ranked_indices(scores, top_n)


_sessions: dict[tuple[str, str], Session] = {}
_sessions_lock = threading.Lock()


def get_backend(requested: str, kind: str) -> Session:
    """Return the cached session for a request model name, creating it lazily."""
    model_name = resolve_model(requested, kind)
    key = (kind, model_name.lower())
    with _sessions_lock:
        session = _sessions.get(key)
        if session is None:
            session = Session(
                model_name,
                kind,
                threads=_threads(),
            )
            _sessions[key] = session
        return session


def reset_sessions() -> None:
    """Drop every cached session. Used by tests and before reconfiguring."""
    with _sessions_lock:
        _sessions.clear()
