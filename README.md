# fastretrieval

**Fast multi-model retrieval runtime: ONNX and GGUF embeddings, reranking, and a declarative model contract**

<!-- Badge Row 1: Status -->
[![CI](https://github.com/n24q02m/fastretrieval/actions/workflows/ci.yml/badge.svg)](https://github.com/n24q02m/fastretrieval/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/n24q02m/fastretrieval/graph/badge.svg?token=M038M651L2)](https://codecov.io/gh/n24q02m/fastretrieval)
[![PyPI](https://img.shields.io/pypi/v/fastretrieval?logo=pypi&logoColor=white)](https://pypi.org/project/fastretrieval/)
[![License: Apache-2.0](https://img.shields.io/github/license/n24q02m/fastretrieval)](LICENSE)

<!-- Badge Row 2: Tech -->
[![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)](#)
[![ONNX Runtime](https://img.shields.io/badge/ONNX_Runtime-005CED?logo=onnx&logoColor=white)](#)
[![Hugging Face](https://img.shields.io/badge/Hugging_Face-FFD21E?logo=huggingface&logoColor=black)](#)
[![semantic-release](https://img.shields.io/badge/semantic--release-e10079?logo=semantic-release&logoColor=white)](https://github.com/python-semantic-release/python-semantic-release)
[![Renovate](https://img.shields.io/badge/renovate-enabled-1A1F6C?logo=renovatebot&logoColor=white)](https://developer.mend.io/)

<!-- BEGIN: AUTO-GENERATED-CROSS-PROMO -->
<details>
  <summary><strong>Sister projects from n24q02m</strong> (click to expand)</summary>

| Project | Tagline | Tag |
|---|---|---|
| [agent-chat-plugin](https://github.com/n24q02m/agent-chat-plugin) | Peer AI agents chat in a shared folder — no human relay, no orchestrator, wor... | Tooling |
| [better-code-review-graph](https://github.com/n24q02m/better-code-review-graph) | Knowledge graph for token-efficient code reviews -- semantic search and call-... | MCP |
| [better-drive](https://github.com/n24q02m/better-drive) | 2-way Google Drive sync with .driveignore filter — rclone engine, Windows tray | Tooling |
| [better-email-mcp](https://github.com/n24q02m/better-email-mcp) | IMAP/SMTP email for AI agents -- read, send, organize folders, and manage att... | MCP |
| [better-godot-mcp](https://github.com/n24q02m/better-godot-mcp) | Composite MCP server for Godot Engine -- 17 composite tools for AI-assisted g... | MCP |
| [better-notion-mcp](https://github.com/n24q02m/better-notion-mcp) | Markdown-first Notion for AI agents -- pages, databases, blocks, and comments... | MCP |
| [better-semantic-release](https://github.com/n24q02m/better-semantic-release) | Drop-in python-semantic-release fork with built-in release-safety guards (orp... | Tooling |
| [better-telegram-mcp](https://github.com/n24q02m/better-telegram-mcp) | Telegram for AI agents -- messages, chats, media, and contacts across both bo... | MCP |
| [better-workspace-mcp](https://github.com/n24q02m/better-workspace-mcp) | Google Workspace MCP server (Docs/Drive/Calendar/Gmail/Sheets/Slides/Tasks/Ch... | MCP |
| [claude-plugins](https://github.com/n24q02m/claude-plugins) | Claude Code plugin marketplace for the n24q02m MCP servers -- install web sea... | Marketplace |
| [imagine-mcp](https://github.com/n24q02m/imagine-mcp) | Image and video understanding + generation for AI agents -- across Gemini, Op... | MCP |
| [jules-task-archiver](https://github.com/n24q02m/jules-task-archiver) | Chrome Extension for bulk operations on Jules tasks via batchexecute API -- a... | Tooling |
| [mcp-core](https://github.com/n24q02m/mcp-core) | Shared foundation for building MCP servers -- Streamable HTTP transport, OAut... | MCP |
| [mnemo-mcp](https://github.com/n24q02m/mnemo-mcp) | Persistent AI memory with hybrid search and embedded sync. Open, free, unlimi... | MCP |
| [fastretrieval](https://github.com/n24q02m/fastretrieval) | Fast multi-model retrieval runtime: ONNX and GGUF embeddings, reranking, and a declarative model contract | Library |
| [skret](https://github.com/n24q02m/skret) | Secrets without the server. | CLI |
| [tacet](https://github.com/n24q02m/tacet) | A self-distilling neuro-symbolic cascade that amortises LLM cost across knowl... | Tooling |
| [web-core](https://github.com/n24q02m/web-core) | Shared web infrastructure package for search, scraping, HTTP security, and st... | Library |
| [wet-mcp](https://github.com/n24q02m/wet-mcp) | Open-source MCP server for AI agents: web search, content extraction, and lib... | MCP |

</details>
<!-- END: AUTO-GENERATED-CROSS-PROMO -->

## What it is

`fastretrieval` is a Python runtime for **multi-model retrieval** with text embeddings and
reranking on ONNX Runtime or GGUF (`llama-cpp-python`) with **no PyTorch dependency**. It
uses a declarative model contract so built-in Qwen3 reference models and custom models can
share the same runtime, and supports Matryoshka (MRL) truncation, instruction-aware queries,
and optional GPU acceleration. It is derived from [fastembed](https://github.com/qdrant/fastembed)
and keeps Qwen3 model names as model identifiers rather than as the package boundary.
Supported runtimes are CPython 3.11, 3.12, 3.13, and 3.14.

## What it does

| Task | Class | Extra needed |
|---|---|---|
| Dense text embedding | `TextEmbedding` | none |
| Sparse embedding (SPLADE) | `SparseTextEmbedding` | none |
| Late interaction (ColBERT) | `LateInteractionTextEmbedding` | none |
| Image embedding | `ImageEmbedding` | `image` |
| Late interaction multimodal (ColPali) | `LateInteractionMultimodalEmbedding` | `image` |
| Cross-encoder rerank | `TextCrossEncoder` | none |
| Generative rerank (yes/no logit) | `TextCrossEncoder` with a YesNo model | none |

## Table of contents

- [What it does](#what-it-does)
- [Features](#features)
- [Supported Models](#supported-models)
- [Installation](#installation)
- [Usage](#usage)
- [Converting your own model](#converting-your-own-model)
- [Configuration](#configuration)
- [Migrating from qwen3-embed](#migrating-from-qwen3-embed)
- [Development](#development)
- [Related Projects](#related-projects)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Model-specific pooling**: Built-in Qwen3 text models use last-token pooling; other models use their declared preprocessing and pooling contract.
- **MRL support**: Qwen3 reference embeddings support truncation from 32 to 1024 dimensions before L2 normalization.
- **Instruction-aware**: Query embedding supports task instructions for better retrieval performance.
- **Causal LM reranking**: Qwen3 rerankers use yes/no logit scoring, producing [0, 1] scores.
- **Multiple backends**: ONNX Runtime and optional GGUF via llama-cpp-python; available formats depend on the selected model.
- **GPU optional, no PyTorch**: Runs on ONNX Runtime or llama-cpp-python -- no heavy ML framework required. Auto-detects GPU (CUDA, DirectML) when available.
- **Multilingual**: Built-in Qwen3 reference models support multi-language inputs.

## Supported Models

The public facades have independent registries. `TextEmbedding()` selects
`n24q02m/Qwen3-Embedding-0.6B-ONNX` (INT8, 1024 dimensions); every other facade
requires an explicit `model_name`. A registry's first entry is not an implicit default.
Qwen3 models are reference models, not a restriction on supported model families.

| Facade | Default selection | Built-in capability |
|:-------|:------------------|:--------------------|
| `TextEmbedding` | Qwen3-Embedding-0.6B-ONNX | Dense text, ONNX or GGUF |
| `TextCrossEncoder` | None; explicit model required | Qwen3 reranking, ONNX or GGUF |
| `SparseTextEmbedding` | None; explicit model required | SPLADE sparse text |
| `LateInteractionTextEmbedding` | None; explicit model required | ColBERT token vectors |
| `ImageEmbedding` | None; explicit model required | Image vectors (`image` extra) |
| `LateInteractionMultimodalEmbedding` | None; explicit model required | ColPali text/image token vectors (`image` extra) |

### Qwen3 ONNX reference models

| Model | Type | Dims | Max Tokens | Size |
|:------|:-----|:-----|:-----------|:-----|
| `n24q02m/Qwen3-Embedding-0.6B-ONNX` | Embedding | 32-1024 (MRL) | 32768 | 573 MB |
| `n24q02m/Qwen3-Embedding-0.6B-ONNX-Q4F16` | Embedding | 32-1024 (MRL) | 32768 | 517 MB |
| `n24q02m/Qwen3-Reranker-0.6B-ONNX` | Reranker | - | 40960 | 573 MB |
| `n24q02m/Qwen3-Reranker-0.6B-ONNX-Q4F16` | Reranker | - | 40960 | 518 MB |
| `n24q02m/Qwen3-Reranker-0.6B-ONNX-YesNo` | Reranker | - | 40960 | 598 MB |

### Qwen3 GGUF reference models (requires the `gguf` extra)

| Model | Type | Dims | Max Tokens | Size |
|:------|:-----|:-----|:-----------|:-----|
| `n24q02m/Qwen3-Embedding-0.6B-GGUF` | Embedding | 32-1024 (MRL) | 32768 | 378 MB |
| `n24q02m/Qwen3-Reranker-0.6B-GGUF` | Reranker | - | 40960 | 378 MB |

### Other built-in ONNX models

| Facade | Model | Output size |
|:-------|:------|:------------|
| `SparseTextEmbedding` | `prithivida/Splade_PP_en_v1` | Sparse vocabulary: 30522 |
| `LateInteractionTextEmbedding` | `colbert-ir/colbertv2.0` | 128 per token |
| `LateInteractionTextEmbedding` | `answerdotai/answerai-colbert-small-v1` | 96 per token |
| `ImageEmbedding` | `Qdrant/clip-ViT-B-32-vision` | 512 |
| `ImageEmbedding` | `Qdrant/resnet50-onnx` | 2048 |
| `ImageEmbedding` | `Qdrant/Unicom-ViT-B-16` | 768 |
| `ImageEmbedding` | `Qdrant/Unicom-ViT-B-32` | 512 |
| `ImageEmbedding` | `jinaai/jina-clip-v1` | 768 |
| `LateInteractionMultimodalEmbedding` | `Qdrant/colpali-v1.3-fp16` | 128 per token |

The sparse registry also reports the deprecated misspelling
`prithvida/Splade_PP_en_v1`; use the canonical `prithivida` spelling above.
Call each facade's `list_supported_models()` for its installed registry metadata
without loading model weights. Register additional dense and reranker models with
`CustomModelSpec` / `CustomRerankerSpec` and the declarative `fastretrieval.contract`
API; custom registration does not change the facade defaults.

### HuggingFace Repos

| Format | Embedding | Reranker |
|:-------|:----------|:--------|
| ONNX | [n24q02m/Qwen3-Embedding-0.6B-ONNX](https://huggingface.co/n24q02m/Qwen3-Embedding-0.6B-ONNX) | [n24q02m/Qwen3-Reranker-0.6B-ONNX](https://huggingface.co/n24q02m/Qwen3-Reranker-0.6B-ONNX) |
| GGUF | [n24q02m/Qwen3-Embedding-0.6B-GGUF](https://huggingface.co/n24q02m/Qwen3-Embedding-0.6B-GGUF) | [n24q02m/Qwen3-Reranker-0.6B-GGUF](https://huggingface.co/n24q02m/Qwen3-Reranker-0.6B-GGUF) |

## Installation

```bash
pip install fastretrieval            # text only
pip install "fastretrieval[image]"   # adds image and ColPali
pip install "fastretrieval[all]"     # adds GGUF backend too

# For GGUF support
pip install fastretrieval[gguf]
```

## Usage

### Text Embedding

```python
from fastretrieval import TextEmbedding

# INT8 (default)
model = TextEmbedding(model_name="n24q02m/Qwen3-Embedding-0.6B-ONNX")

# Q4F16 (smaller, slightly less accurate)
model = TextEmbedding(model_name="n24q02m/Qwen3-Embedding-0.6B-ONNX-Q4F16")

# GGUF (requires: pip install fastretrieval[gguf])
model = TextEmbedding(model_name="n24q02m/Qwen3-Embedding-0.6B-GGUF")

documents = [
    "Qwen3 is a multilingual embedding model.",
    "ONNX Runtime enables fast CPU inference.",
]

embeddings = list(model.embed(documents))
# Each embedding: numpy array of shape (1024,), L2-normalized

# Matryoshka Representation Learning (MRL) -- truncate to smaller dims
embeddings_256 = list(model.embed(documents, dim=256))
# Each embedding: numpy array of shape (256,), L2-normalized

# Query with instruction (for retrieval tasks)
queries = list(
    model.query_embed(
        ["What is Qwen3?"],
        task="Given a question, retrieve relevant passages",
    )
)
```

### Reranking

```python
from fastretrieval import TextCrossEncoder

reranker = TextCrossEncoder(model_name="n24q02m/Qwen3-Reranker-0.6B-ONNX")

# YesNo variant: ~10x less RAM (~598MB vs ~12GB at inference)
# reranker = TextCrossEncoder(model_name="n24q02m/Qwen3-Reranker-0.6B-ONNX-YesNo")

query = "What is Qwen3?"
documents = [
    "Qwen3 is a series of large language models by Alibaba.",
    "The weather today is sunny.",
    "Qwen3-Embedding supports multilingual text embedding.",
]

scores = list(reranker.rerank(query, documents))
# scores: list of float in [0, 1], higher = more relevant

# Or rerank pairs directly
pairs = [
    ("What is AI?", "Artificial intelligence is a branch of computer science."),
    ("What is ML?", "Machine learning is a subset of AI."),
]
pair_scores = list(reranker.rerank_pairs(pairs))
```

#### Reranker determinism

Reranker scores are **batch-invariant**: the score of a `(query, document)` pair
does not depend on batch size or the other documents scored in the same call.
ONNX reranker variants are scored one sequence at a time (no padding), which keeps
RoPE positions correct regardless of batch composition.

### Custom models (bring your own)

Qwen3 models are built-in reference models, but any ONNX-able embedding model can be
registered and then loaded by id. Use `CustomModelSpec` with one of the four
output shapes: `CLS`/`MEAN` (bert-bi), `LAST_TOKEN` (causal), or `DISABLED` (raw).

```python
from fastretrieval import CustomModelSpec, TextEmbedding

# Multilingual (incl. Vietnamese) + code, CLS-pooled, 768-dim
CustomModelSpec(
    model_id="onnx-community/gte-multilingual-base",
    hf="onnx-community/gte-multilingual-base",
    model_file="onnx/model_quantized.onnx",
    dim=768,
    pooling="CLS",
    normalization=True,
).register()

model = TextEmbedding("onnx-community/gte-multilingual-base")
embeddings = list(model.embed(["xin chào", "def add(a, b): return a + b"]))
```

Other verified examples: `bge-m3` (`pooling="CLS"`, `dim=1024`), `EmbeddingGemma-300m`
(`pooling="MEAN"`, `dim=768`). MRL truncation (`embed(..., dim=256)`) works for custom
models whose vectors are Matryoshka-trained. Custom models are scored per-row, so —
like the built-in INT8 reranker — their scores are batch-invariant by construction.

A BYO **reranker** registers the same way with `CustomRerankerSpec`. Any standard ONNX
cross-encoder (a single relevance logit per pair — `bge-reranker`, `gte-reranker`,
`ms-marco`, `jina-reranker`) works; there is no `dim`/`pooling` to set:

```python
from fastretrieval import CustomRerankerSpec, TextCrossEncoder

CustomRerankerSpec(
    model_id="onnx-community/gte-multilingual-reranker-base",
    hf="onnx-community/gte-multilingual-reranker-base",
    model_file="onnx/model_quantized.onnx",
).register()

encoder = TextCrossEncoder("onnx-community/gte-multilingual-reranker-base")
scores = list(encoder.rerank("xin chào", ["tài liệu A", "tài liệu B"]))
```

PyTorch-only models can be converted first (in a throwaway env, since the export
deps don't co-resolve with the lean runtime pins):

```python
# pip install "optimum-onnx[onnxruntime]" torch transformers onnx
from fastretrieval.export import export_to_onnx

export_to_onnx("intfloat/multilingual-e5-base", "./e5-onnx")
```

## Converting your own model

`fastretrieval` does not ship a closed model zoo. Point the converter at a model
family in the support matrix and it produces an artifact with a declarative
contract manifest. Unsupported architectures fail closed instead of producing a
misleading artifact.

The conversion dependencies (`torch`, `transformers`, `optimum-onnx`, and the
ONNX Runtime quantizer stack including `onnx-ir`) are deliberately not runtime
dependencies. Run conversion in a throwaway environment:

```bash
uv run --with-requirements fastretrieval/convert/requirements.txt \
  python -m fastretrieval.convert onnx intfloat/multilingual-e5-base \
  --out ./e5 --pooling mean --normalize

uv run --with-requirements fastretrieval/convert/requirements.txt \
  python -m fastretrieval.convert verify ./e5 \
  --source intfloat/multilingual-e5-base
```

`verify` validates the manifest, loads every ONNX variant through ONNX Runtime,
and compares the converted outputs with the original model on the same probes.
The `card` command writes a model card only after the manifest and artifact
formats pass validation. GGUF conversion additionally requires a built
`llama.cpp` checkout and `--llama-cpp` (or `LLAMA_CPP_HOME`).

Large models can run in the optional Modal backend. It mounts local sources and
downloads the completed artifact back to the requested output directory; it does
not publish to a model hub:

```bash
uv run --with-requirements fastretrieval/convert/requirements.txt \
  python -m fastretrieval.convert onnx Qwen/Qwen3-Embedding-0.6B \
  --out ./qwen3 --pooling last_token --normalize --backend modal
```

The first acceptance profiles are Qwen3 and BERT-family text models. Adding a
model family requires its architecture, task, modality, pooling, normalization,
and output-shape profile plus parity tests; the model name alone is never enough.

## Configuration

### Environment variables

| Variable | Purpose |
|:---------|:--------|
| `FASTRETRIEVAL_CACHE_PATH` | Override the model cache directory. |
| `FASTRETRIEVAL_MAX_INPUT_LENGTH` | Override the maximum accepted input length. |
| `QWEN3_EMBED_CACHE_PATH` | Deprecated compatibility alias for `FASTRETRIEVAL_CACHE_PATH`. |
| `QWEN3_EMBED_MAX_INPUT_LENGTH` | Deprecated compatibility alias for `FASTRETRIEVAL_MAX_INPUT_LENGTH`. |

The `FASTRETRIEVAL_*` names take precedence when both names are set. The deprecated
`QWEN3_EMBED_*` names remain readable and emit a `DeprecationWarning` so existing
configurations do not silently stop working.

### Programmatic cache location

Consumers that need to inspect or clear the model cache should use the public
helper instead of importing an internal module:

```python
from fastretrieval import define_cache_dir

cache_dir = define_cache_dir()
```

`define_cache_dir()` honors `FASTRETRIEVAL_CACHE_PATH` first and the deprecated
`QWEN3_EMBED_CACHE_PATH` alias second. An explicit path argument overrides both.

### GPU Acceleration

Both ONNX and GGUF backends auto-detect GPU when available (`Device.AUTO` is the default).

#### ONNX

Requires `onnxruntime-gpu` (CUDA) or `onnxruntime-directml` (Windows) instead of `onnxruntime`:

```bash
pip install onnxruntime-gpu  # NVIDIA CUDA
# or
pip install onnxruntime-directml  # Windows AMD/Intel/NVIDIA
```

```python
from fastretrieval import TextEmbedding, Device

# Auto-detect GPU (default)
model = TextEmbedding(model_name="n24q02m/Qwen3-Embedding-0.6B-ONNX")

# Force CPU
model = TextEmbedding(model_name="n24q02m/Qwen3-Embedding-0.6B-ONNX", cuda=Device.CPU)

# Force CUDA
model = TextEmbedding(model_name="n24q02m/Qwen3-Embedding-0.6B-ONNX", cuda=Device.CUDA)
```

#### GGUF

GPU is handled by `llama-cpp-python`. The default `pip install fastretrieval[gguf]` is CPU-only.
For CUDA GPU support, build with:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install fastretrieval[gguf]
```

```python
from fastretrieval import TextEmbedding, Device

# Auto-detect GPU (default, offloads all layers)
model = TextEmbedding(model_name="n24q02m/Qwen3-Embedding-0.6B-GGUF")

# Force CPU only
model = TextEmbedding(model_name="n24q02m/Qwen3-Embedding-0.6B-GGUF", cuda=Device.CPU)
```

## Development

```bash
uv sync --group dev                              # Install dev dependencies
uv run ruff check .                              # Lint
uv run ruff format --check .                     # Format check
uv run ty check                                  # Type check
uv run pytest                                    # All tests (integration tests download ~1.2 GB)
uv run pytest -m "not integration" --tb=short    # Unit tests only (CI default)

# Shortcuts (optional, via mise): mise run setup / lint / test / fix
```

## Migrating from qwen3-embed

The library was previously published as `qwen3-embed`. New releases use the
`fastretrieval` distribution and import package; Qwen3 model identifiers remain unchanged.

### Package and import names

```bash
# Before
pip install qwen3-embed[gguf]

# After
pip install fastretrieval[gguf]
```

```python
# Before
from qwen3_embed import TextEmbedding, TextCrossEncoder

# After
from fastretrieval import TextEmbedding, TextCrossEncoder
```

The public API is unchanged. The old environment variable names remain supported as
deprecated compatibility aliases:

| Old variable | New variable |
|:-------------|:-------------|
| `QWEN3_EMBED_CACHE_PATH` | `FASTRETRIEVAL_CACHE_PATH` |
| `QWEN3_EMBED_MAX_INPUT_LENGTH` | `FASTRETRIEVAL_MAX_INPUT_LENGTH` |

The old names still work and emit a `DeprecationWarning`; when both names are set, the
`FASTRETRIEVAL_*` value wins. Existing `qwen3-embed` releases remain on PyPI and continue
to receive security fixes while consumers migrate.

## Related Projects

- [wet-mcp](https://github.com/n24q02m/wet-mcp) -- MCP web search server with vector-based docs search, uses fastretrieval for local embedding
- [mnemo-mcp](https://github.com/n24q02m/mnemo-mcp) -- MCP memory server with semantic search powered by fastretrieval
- [better-code-review-graph](https://github.com/n24q02m/better-code-review-graph) -- Knowledge graph for code reviews, uses fastretrieval for local ONNX embedding
- [modalcom-ai-workers](https://github.com/n24q02m/modalcom-ai-workers) -- GPU-serverless workers that convert Qwen3 models to ONNX/GGUF format

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0 -- See [LICENSE](LICENSE). Original fastembed by [Qdrant](https://github.com/qdrant/fastembed).
