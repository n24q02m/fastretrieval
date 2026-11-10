"""HTTP API: ``/embed``, ``/rerank``, ``/health`` and ``/models``.

Serves the retrieval runtime over HTTP so several clients share one model
instance instead of loading their own. Model artifacts download from Hugging
Face into the user cache lazily on first use. ``*-GGUF`` names need the
``fastretrieval[gguf]`` extra; the HTTP stack itself is
``fastretrieval[server]``.
"""

from __future__ import annotations

import asyncio
import logging

from fastretrieval.server._extra import require_server_extra

# Must run before the extra's own imports below: fail with the install hint
# instead of a bare ModuleNotFoundError on a base fastretrieval install.
require_server_extra()

from fastapi import FastAPI, HTTPException, status  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from fastretrieval.server import __version__, registry  # noqa: E402
from fastretrieval.server.registry import DIM_MAX  # noqa: E402

logger = logging.getLogger(__name__)


class EmbedRequest(BaseModel):
    model: str = "qwen3-0.6b"
    input: list[str]
    dims: int = DIM_MAX


class EmbedResponse(BaseModel):
    data: list[list[float]]
    model: str
    dims: int


class RerankRequest(BaseModel):
    model: str = "qwen3-rerank-0.6b"
    query: str
    documents: list[str]
    top_n: int | None = Field(default=None, ge=1)


class RerankResult(BaseModel):
    index: int
    relevance_score: float


class RerankResponse(BaseModel):
    results: list[RerankResult]
    model: str


class HealthResponse(BaseModel):
    status: str
    version: str


class ModelsResponse(BaseModel):
    embed: dict[str, str]
    rerank: dict[str, str]


app = FastAPI(title="fastretrieval-server", version=__version__)


def _backend_error(exc: Exception) -> HTTPException:
    """Map backend failures to client-visible HTTP errors."""
    if isinstance(exc, ValueError):
        # Unsupported model name or out-of-range dims.
        return HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc))
    if isinstance(exc, ImportError):
        # Model class needs an opt-in runtime (e.g. llama-cpp-python for GGUF).
        return HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc))
    logger.exception("embedding backend failed")
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail=f"embedding backend failed: {type(exc).__name__}: {exc}",
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@app.get("/models", response_model=ModelsResponse)
async def models() -> ModelsResponse:
    """Advertise the public alias table and the canonical names it maps to."""
    return ModelsResponse(embed=dict(registry.EMBED_MODELS), rerank=dict(registry.RERANK_MODELS))


@app.post("/embed", response_model=EmbedResponse)
async def embed(req: EmbedRequest) -> EmbedResponse:
    backend = registry.get_backend(req.model, "embed")
    try:
        vectors = await asyncio.to_thread(backend.embed, req.input, req.dims)
    except Exception as exc:  # mapped below; FastAPI re-raises HTTPException unchanged
        raise _backend_error(exc) from exc
    resolved = registry.resolve_model(req.model, "embed")
    return EmbedResponse(
        data=vectors, model=resolved, dims=len(vectors[0]) if vectors else req.dims
    )


@app.post("/rerank", response_model=RerankResponse)
async def rerank(req: RerankRequest) -> RerankResponse:
    backend = registry.get_backend(req.model, "rerank")
    try:
        ranked = await asyncio.to_thread(backend.rerank, req.query, req.documents, req.top_n)
    except Exception as exc:
        raise _backend_error(exc) from exc
    resolved = registry.resolve_model(req.model, "rerank")
    return RerankResponse(
        results=[RerankResult(index=i, relevance_score=s) for i, s in ranked],
        model=resolved,
    )
