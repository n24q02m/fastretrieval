# AGENTS.md - fastretrieval

Fast multi-model retrieval runtime: ONNX and GGUF embeddings, reranking, and a declarative model contract. Python >= 3.11 (tested 3.11-3.14), uv.

## Package Identity

- Distribution: `fastretrieval`; import package: `fastretrieval`.
- Install the runtime with `pip install fastretrieval`; add `[gguf]` for GGUF support.
- New environment variables: `FASTRETRIEVAL_CACHE_PATH` and `FASTRETRIEVAL_MAX_INPUT_LENGTH`.
- Deprecated compatibility aliases: `QWEN3_EMBED_CACHE_PATH` and `QWEN3_EMBED_MAX_INPUT_LENGTH`.
  The new names win when both are set; old names remain readable and emit `DeprecationWarning`.
- Qwen3 adapter/module/test names remain model-specific identifiers and must not be renamed as
  part of package identity work.

## Build / Lint / Test Commands

```bash
uv sync --group dev                # Install dependencies
uv build                           # Build package (hatchling)
uv run ruff check .                # Lint
uv run ruff format --check .       # Format check
uv run ruff format .               # Format fix
uv run ruff check --fix .          # Lint fix
uv run ty check                    # Type check (Astral ty)

# Tests (integration tests download ~1.2 GB model)
uv run pytest                                       # All tests including integration
uv run pytest -m "not integration" --tb=short       # Unit tests only (CI default)

# Run a single test file
uv run pytest tests/test_utils.py

# Run a single test function
uv run pytest tests/test_utils.py::test_function_name -v

# Mise shortcuts
mise run setup     # Full dev environment setup
mise run lint      # ruff check + ruff format --check + ty check
mise run test      # pytest
mise run fix       # ruff check --fix + ruff format
```

### Pytest Configuration

- `testpaths = ["tests"]`, `pythonpath = ["."]`
- Integration marker: `@pytest.mark.integration` (requires model downloads: ONNX ~1.2 GB, Q4F16 ~1 GB, GGUF ~756 MB)
- Integration test files: `test_integration.py` (ONNX), `test_integration_q4f16.py`, `test_integration_gguf.py`
- CI runs: `uv run pytest -m "not integration" --tb=short`

## Code Style

### Formatting (Ruff)

- **Line length**: 99
- **Quotes**: Double quotes
- **Indent**: 4 spaces
- **Target**: Python 3.13

### Ruff Rules

`select = ["E", "F", "I", "UP", "B", "SIM"]`, `ignore = ["E501"]`

- `I` = isort, `UP` = pyupgrade, `B` = bugbear, `SIM` = simplify

### Type Checker (ty)

Custom rules in `[tool.ty.rules]` (pyproject.toml):
- `unresolved-import = "warn"` (llama_cpp is optional)
- `possibly-missing-attribute = "warn"` (onnxruntime incomplete stubs)
- `invalid-argument-type = "warn"`, `unresolved-attribute = "warn"`, `not-subscriptable = "warn"`, `invalid-assignment = "warn"`, `call-non-callable = "warn"`

### Import Ordering (isort via Ruff)

1. Standard library (`import json`, `from pathlib import Path`, `from typing import Any`)
2. Third-party (`import numpy as np`, `from loguru import logger`, `from tokenizers import Tokenizer`)
3. Local (`from fastretrieval.common.types import ...`)

```python
import json
import os
from pathlib import Path

import numpy as np
from loguru import logger

from fastretrieval.common.types import PathInput, Device
```

### Type Hints

- Full type hints everywhere: parameters, return types, variables
- **Python 3.12+ type alias syntax**: `type PathInput = str | Path`
- **Python 3.12+ generics**: `class ModelManagement[T: BaseModelDescription]:`, `def iter_batch[T](...)`
- Union types: `str | None` (not `Optional`), `list[str]` (not `List`)
- `py.typed` marker file present

### Naming Conventions

| Element            | Convention       | Example                             |
|--------------------|------------------|-------------------------------------|
| Functions/methods  | snake_case       | `last_token_pool`, `load_onnx_model` |
| Private methods    | `_snake_case`    | `_preprocess_onnx_input`, `_get_model_description` |
| Classes            | PascalCase       | `TextEmbedding`, `ModelManagement`  |
| Constants          | UPPER_SNAKE_CASE | `METADATA_FILE`, `EXPOSED_SESSION_OPTIONS` |
| Modules/packages   | snake_case       | `model_management.py`, `cross_encoder` |
| Enums              | PascalCase class | `Device.CPU`, `PoolingType.LAST_TOKEN` |

### Error Handling

- `ValueError` for input/config validation errors
- `PermissionError` for authentication failures
- `raise ... from e` for exception chaining
- `assert` for internal invariants
- `warnings.warn()` with `UserWarning`/`RuntimeWarning` for non-fatal issues
- `loguru.logger` for logging (not stdlib `logging`, except `parallel_processor.py`)
- try/except with `pass` for optional cache loading

### File Organization

```
fastretrieval/                    # Main package (not src layout)
  __init__.py                     # Public API: TextEmbedding, TextCrossEncoder
  py.typed                        # PEP 561 marker
  parallel_processor.py           # Multiprocessing worker pool
  common/                         # Shared utilities
    types.py                      # Type aliases, Device enum
    utils.py                      # Pooling, normalize, batching helpers
    model_description.py          # Dataclasses for model metadata
    model_management.py           # Model download/cache (HF Hub, GCS)
    onnx_model.py                 # Base ONNX model class
  text/                           # Embedding module
    text_embedding.py             # Public facade
    onnx_embedding.py, qwen3_embedding.py, gguf_embedding.py, ...
  rerank/cross_encoder/           # Reranking module
    text_cross_encoder.py         # Public facade
    qwen3_cross_encoder.py, gguf_cross_encoder.py, ...
tests/
  test_utils.py, test_pooling.py, test_qwen3_embedding.py, ...
  test_integration.py             # Requires real model download
```

### Documentation

- Google-style docstrings with `Args:`, `Returns:`, `Raises:` sections
- Not all functions have docstrings -- only public/complex methods

### Commits

Conventional Commits: `type(scope): message`. Automated semantic release via PSR v10.

### Pre-commit Hooks

1. Ruff lint (`--fix --target-version=py313`) + format
2. pytest (`-m "not integration" --tb=short -q`)

## Converter

The optional conversion stack lives in `fastretrieval/convert/requirements.txt` and is
not part of the runtime lock. Run it with `uv run --with-requirements ...`; do not add
PyTorch, Optimum ONNX, or Transformers to the runtime dependencies just to make conversion
convenient. The ONNX path requires explicit pooling and normalization metadata, writes
the versioned contract manifest, validates every exported variant with ONNX Runtime,
and only then promotes the artifact directory. `python -m fastretrieval.convert verify`
must be run before publishing an artifact. `--backend modal` is optional; local is the
default and GGUF additionally requires a built `llama.cpp` checkout.
