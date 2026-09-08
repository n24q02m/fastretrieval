# fastretrieval Handover

## Product boundary

`fastretrieval` is a Python retrieval runtime, not an MCP server. It provides dense, sparse, late-interaction, multimodal, and cross-encoder facades over ONNX Runtime and optional GGUF backends. Wet, Mnemo, and Better Code Review Graph consume its public embedding and reranking contracts; they own transport, authentication, and deployment.

Package identity is `fastretrieval` for both distribution and import. Qwen3 names remain model identifiers. The deprecated `QWEN3_EMBED_*` environment variables remain readable and warn; `FASTRETRIEVAL_*` values take precedence.

## Current operation

- Python: CPython 3.11–3.14.
- Runtime setup: `uv sync --group dev`.
- Unit gate: `uv run pytest -m "not integration" --tb=short`.
- Full tests include model downloads; run only when the change requires them.
- Lint/type checks: `uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check`.
- Build: `uv build`.
- Converter: use `uv run --with-requirements fastretrieval/convert/requirements.txt`; validate artifacts with `python -m fastretrieval.convert verify` before publication.
- Default dense model: `n24q02m/Qwen3-Embedding-0.6B-ONNX`.
- Other facades require an explicit model identifier; inspect `list_supported_models()` rather than assuming a shared default.

## Source map

- `fastretrieval/text/`: dense embedding facade and backends.
- `fastretrieval/models/sparse/`: SPLADE sparse embeddings.
- `fastretrieval/models/late_interaction/`: ColBERT token embeddings.
- `fastretrieval/models/late_interaction_multimodal/`: ColPali.
- `fastretrieval/rerank/cross_encoder/`: cross-encoder and YesNo reranking.
- `fastretrieval/contract/`: versioned model artifact contract.
- `fastretrieval/convert/`: isolated ONNX/GGUF conversion tooling.
- `tests/`: deterministic unit and contract tests; integration tests are download-marked.

## Recent correctness work

SPLADE reduction masks padded tokens before max/log processing and uses bounded in-place operations. ColBERT document post-processing masks skipped/padded tokens and normalizes active vectors with a clamped L2 norm. Modal artifact extraction rejects absolute and parent traversal paths. Focused tests cover score parity, token masking, zero-norm finiteness, and traversal rejection.

## Consumer contract

Consumers should depend on the published `fastretrieval` package and import public facades from `fastretrieval`. Do not add a `qwen3-embed` dependency or invent a fallback import. Keep model selection explicit for non-dense facades. Consumer changes must verify the exact producer artifact before adoption; a source merge alone is not runtime evidence.

## Modernization map

1. Keep one implementation of each model contract in this package; expose adapters only through the existing public facades.
2. Preserve the declarative manifest fields in `fastretrieval/contract/` when adding model families. A model name alone does not define pooling, normalization, modality, or output shape.
3. Keep conversion dependencies isolated from runtime dependencies and promote only artifacts that pass manifest and ONNX Runtime verification.
4. Add focused behavioral coverage for any post-processing optimization before merging; benchmark claims do not replace score-parity tests.
5. Treat package publication, consumer adoption, and live consumer behavior as separate evidence checkpoints.

## Rollback and triage

For a regression, identify the exact model facade and artifact contract first, then reproduce with a local fixture or an existing model cache. Revert the smallest owning commit or pin the last verified package artifact; do not change consumer authentication or user-owned runtime profiles. For conversion failures, inspect the manifest and rerun `verify` before rebuilding. For consumer failures, compare producer version, model identifier, dimensions, normalization, and tokenizer contract.

## Scope boundaries

This repository has no MCP protocol, relay, clean-state, or client Test B surface. Do not add those checks. Do not depend on OCI VM infrastructure, paid providers, scheduled synchronization, or user-owned OMP configuration for package operation.
