# CLAUDE.md - qwen3-embed

Thu vien embedding va reranking Qwen3 qua ONNX Runtime va GGUF.
Python >= 3.11 (ho tro 3.11-3.14), uv, hatchling. KHONG phai src layout -- package tai `qwen3_embed/`.
Fork cua fastembed (Qdrant), chi giu Qwen3 models. License: Apache-2.0.

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
qwen3_embed/                      # Main package (KHONG phai src layout)
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

PSR v10 (workflow_dispatch) -> PyPI. Khong co Docker (la library, khong phai server).

## Luu y

- KHONG phai src layout: package truc tiep tai `qwen3_embed/`, khong phai `src/qwen3_embed/`.
- `requires-python = ">=3.11"` -- ho tro rong hon cac project khac (3.11-3.14).
- Optional dependency: `pip install qwen3-embed[gguf]` cho llama-cpp-python.
- GPU auto-detect: ONNX (onnxruntime-gpu/directml), GGUF (llama-cpp-python CUDA build).
- Last-token pooling (khong phai mean pooling) + MRL support (truncate 32-1024 dims).
- YesNo reranker variant: ~10x it RAM (~598MB vs ~12GB).
- Model cache: HuggingFace Hub cache directory.
- Pre-commit: ruff lint + format, pytest unit only.
- Secrets: skret SSM namespace `/qwen3-embed/prod` (region `ap-southeast-1`)
