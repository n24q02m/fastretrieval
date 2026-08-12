# CLAUDE.md - fastretrieval

Fast multi-model retrieval runtime: ONNX and GGUF embeddings, reranking, and a declarative model contract.
Python >= 3.11 (ho tro 3.11-3.14), uv, hatchling. KHONG phai src layout -- package tai `fastretrieval/`.
Derived from fastembed (Qdrant), voi Qwen3 adapters la built-in reference models; custom model
contracts mo rong ra ngoai family nay. License: Apache-2.0.

## Package identity

- Distribution/import: `fastretrieval`.
- Install: `pip install fastretrieval`; optional GGUF: `pip install fastretrieval[gguf]`.
- Env moi: `FASTRETRIEVAL_CACHE_PATH` va `FASTRETRIEVAL_MAX_INPUT_LENGTH`.
- Env tuong thich deprecated: `QWEN3_EMBED_CACHE_PATH` va `QWEN3_EMBED_MAX_INPUT_LENGTH`.
  Ten moi duoc uu tien neu ca hai duoc dat; ten cu van doc duoc va phat `DeprecationWarning`.
- Qwen3 model identifiers va adapter filenames giu nguyen vi day la model names.

## Commands

```bash
# Setup
uv sync --group dev

# Lint & Type check
uv run ruff check .
uv run ruff format --check .
uv run ty check

# Fix
uv run ruff check --fix .
uv run ruff format .

# Test (unit only -- CI default)
uv run pytest -m "not integration" --tb=short
uv run pytest                                       # tat ca, bao gom integration
uv run pytest tests/test_utils.py -v                # single file

# Build
uv build

# Mise shortcuts
mise run setup     # full dev setup
mise run lint      # ruff check + format check + ty check
mise run test      # pytest
mise run fix       # ruff fix + format
```

## Pytest

- `testpaths = ["tests"]`, `pythonpath = ["."]`
- Integration marker: `@pytest.mark.integration` (can download model: ONNX ~1.2GB, Q4F16 ~1GB, GGUF ~756MB)
- CI chi chay unit tests: `-m "not integration"`
- Snapshot testing: syrupy

## Cau truc thu muc

```
fastretrieval/                    # Main package (KHONG phai src layout)
  __init__.py                     # Public API: TextEmbedding, TextCrossEncoder
  py.typed                        # PEP 561 marker
  parallel_processor.py           # Multiprocessing worker pool
  common/                         # types, utils, model_description, model_management, onnx_model
  text/                           # Embedding: text_embedding.py (facade), onnx/qwen3/gguf variants
  rerank/cross_encoder/           # Reranking: text_cross_encoder.py (facade), qwen3/gguf variants
tests/
  test_utils.py, test_pooling.py, test_qwen3_embedding.py, ...
  test_integration.py             # Can real model download
```

## Models

| Model | Type | Size |
|-------|------|------|
| `n24q02m/Qwen3-Embedding-0.6B-ONNX` | Embedding | 573 MB |
| `n24q02m/Qwen3-Embedding-0.6B-ONNX-Q4F16` | Embedding | 517 MB |
| `n24q02m/Qwen3-Reranker-0.6B-ONNX` | Reranker | 573 MB |
| `n24q02m/Qwen3-Reranker-0.6B-ONNX-Q4F16` | Reranker | 518 MB |
| `n24q02m/Qwen3-Reranker-0.6B-ONNX-YesNo` | Reranker | 598 MB |
| `n24q02m/Qwen3-Embedding-0.6B-GGUF` | Embedding | 378 MB |
| `n24q02m/Qwen3-Reranker-0.6B-GGUF` | Reranker | 378 MB |

## Code conventions

- Ruff: line-length 99 (khac 88 cua cac project khac), double quotes
- Rules: `["E", "F", "I", "UP", "B", "SIM"]` (co SIM, khong co W, C4)
- Python 3.12+ syntax: `type PathInput = str | Path`, `class Foo[T: Base]:`
- ty: nhieu rules o muc `warn` (khong phai ignore) vi optional deps va incomplete stubs
- Error handling: ValueError, PermissionError, `raise ... from e`, `warnings.warn()`
- Logging: `loguru` (tru `parallel_processor.py` dung stdlib logging)

## CD Pipeline

PSR v10 (workflow_dispatch) -> PyPI. Dockerfile cung cap stdio/http targets.

## Luu y

- KHONG phai src layout: package truc tiep tai `fastretrieval/`, khong phai `src/fastretrieval/`.
- `requires-python = ">=3.11"` -- ho tro rong hon cac project khac (3.11-3.14).
- Optional dependency: `pip install fastretrieval[gguf]` cho llama-cpp-python.
- GPU auto-detect: ONNX (onnxruntime-gpu/directml), GGUF (llama-cpp-python CUDA build).
- Last-token pooling (khong phai mean pooling) + MRL support (truncate 32-1024 dims).
- YesNo reranker variant: ~10x it RAM (~598MB vs ~12GB).
- Model cache: HuggingFace Hub cache directory.
- Pre-commit: ruff lint + format, pytest unit only.
- Secrets: dung namespace SSM/skret do repo cau hinh (region `ap-southeast-1`); khong hardcode credentials.

## Converter

The converter stack is isolated in `fastretrieval/convert/requirements.txt`; run it with
`uv run --with-requirements` instead of adding heavy export dependencies to the runtime
lock. ONNX conversion requires explicit pooling/normalization metadata, writes the
versioned manifest, validates every variant with ONNX Runtime, then promotes the output.
Run `python -m fastretrieval.convert verify` before publishing. Local execution is the
default; `--backend modal` is optional, and GGUF requires a built `llama.cpp` checkout.
