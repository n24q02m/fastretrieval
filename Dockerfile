# syntax=docker/dockerfile:1
# Container image for the fastretrieval model runtime and its shared HTTP
# surface (``fastretrieval.server``: /embed, /rerank, /health, /models).
#
#   docker build -t fastretrieval .
#   docker run --rm -p 9800:9800 -v fastretrieval-models:/data fastretrieval
#
# Model artifacts download from Hugging Face on first request, so /data must be
# a volume to survive a container restart: a cold cache re-downloads ~1.2 GB.

# =============================================================================
# Stage 1: builder -- resolve the locked dependency set and build the wheel
# =============================================================================
# Same Python minor as the runtime stage: a .venv built on one minor is not
# portable to another (binary wheels, ABI tags).
FROM python:3.13-slim-bookworm AS builder

# The uv binary only; its own bundled interpreter is older than this base.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependency layer: manifest + lockfile only, so editing source never
# re-resolves the dependency graph. --no-install-project keeps the project
# itself out of this layer for the same reason.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra server --no-install-project

# Project layer: --no-editable installs a real copy into site-packages, so the
# runtime stage needs no source tree at all.
COPY fastretrieval ./fastretrieval
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra server --no-editable

# =============================================================================
# Stage 2: runtime
# =============================================================================
FROM python:3.13-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    FASTRETRIEVAL_CACHE_PATH=/data/models

WORKDIR /app

# Non-root runtime user; /data owns the model cache volume.
RUN groupadd --system --gid 10001 app \
    && useradd --system --uid 10001 --gid app --home-dir /home/app --create-home app \
    && mkdir -p /data/models \
    && chown -R app:app /data /home/app

COPY --from=builder --chown=app:app /app/.venv /app/.venv

VOLUME /data
USER app
EXPOSE 9800

# The server binds 127.0.0.1 by default; CMD widens it for the container. Keep
# this port in step with CMD -- it is the probe target.
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD ["python", "-c", "import sys, urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:9800/health', timeout=5).status == 200 else 1)"]

ENTRYPOINT ["python", "-m", "fastretrieval.server"]
CMD ["--host", "0.0.0.0", "--port", "9800"]
