# Qwen3 Embed

**Lightweight Qwen3 text embedding and reranking via ONNX Runtime and GGUF**

<!-- Badge Row 1: Status -->
[![CI](https://github.com/n24q02m/qwen3-embed/actions/workflows/ci.yml/badge.svg)](https://github.com/n24q02m/qwen3-embed/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/n24q02m/qwen3-embed/graph/badge.svg?token=M038M651L2)](https://codecov.io/gh/n24q02m/qwen3-embed)
[![PyPI](https://img.shields.io/pypi/v/qwen3-embed?logo=pypi&logoColor=white)](https://pypi.org/project/qwen3-embed/)
[![License: Apache-2.0](https://img.shields.io/github/license/n24q02m/qwen3-embed)](LICENSE)

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
| [qwen3-embed](https://github.com/n24q02m/qwen3-embed) | Lightweight Qwen3 text embedding and reranking via ONNX Runtime and GGUF | Library |
| [skret](https://github.com/n24q02m/skret) | Secrets without the server. | CLI |
| [tacet](https://github.com/n24q02m/tacet) | A self-distilling neuro-symbolic cascade that amortises LLM cost across knowl... | Tooling |
| [web-core](https://github.com/n24q02m/web-core) | Shared web infrastructure package for search, scraping, HTTP security, and st... | Library |
| [wet-mcp](https://github.com/n24q02m/wet-mcp) | Open-source MCP server for AI agents: web search, content extraction, and lib... | MCP |

</details>
<!-- END: AUTO-GENERATED-CROSS-PROMO -->

## What it is

`qwen3-embed` is a lightweight Python library for **text embedding** and **reranking** with
Qwen3 0.6B models. It runs on ONNX Runtime or GGUF (`llama-cpp-python`) with **no PyTorch
dependency**, supports Matryoshka (MRL) truncation, instruction-aware queries, and optional
GPU acceleration. It is a trimmed fork of [fastembed](https://github.com/qdrant/fastembed)
that keeps only the Qwen3 models, and any ONNX-able model can be registered as a custom model.

## Table of contents

- [Features](#features)
- [Supported Models](#supported-models)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Development](#development)
- [Related Projects](#related-projects)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Last-token pooling**: Uses the final token representation (with left-padding) instead of mean pooling.
- **MRL support**: Matryoshka Representation Learning allows truncating embeddings to any dimension from 32 to 1024 while preserving quality.
- **Instruction-aware**: Query embedding supports task instructions for better retrieval performance.
- **Causal LM reranking**: Reranker uses yes/no logit scoring via causal language model, producing calibrated [0, 1] scores.
- **Multiple backends**: ONNX Runtime (INT8, Q4F16) and GGUF (Q4_K_M via llama-cpp-python).
- **GPU optional, no PyTorch**: Runs on ONNX Runtime or llama-cpp-python -- no heavy ML framework required. Auto-detects GPU (CUDA, DirectML) when available.
- **Multilingual**: Both models support multi-language inputs.

## Supported Models

### ONNX (default)

| Model | Type | Dims | Max Tokens | Size |
|:------|:-----|:-----|:-----------|:-----|
| `n24q02m/Qwen3-Embedding-0.6B-ONNX` | Embedding | 32-1024 (MRL) | 32768 | 573 MB |
| `n24q02m/Qwen3-Embedding-0.6B-ONNX-Q4F16` | Embedding | 32-1024 (MRL) | 32768 | 517 MB |
| `n24q02m/Qwen3-Reranker-0.6B-ONNX` | Reranker | - | 40960 | 573 MB |
| `n24q02m/Qwen3-Reranker-0.6B-ONNX-Q4F16` | Reranker | - | 40960 | 518 MB |
| `n24q02m/Qwen3-Reranker-0.6B-ONNX-YesNo` | Reranker | - | 40960 | 598 MB |

### GGUF (optional, requires `llama-cpp-python`)

| Model | Type | Dims | Max Tokens | Size |
|:------|:-----|:-----|:-----------|:-----|
| `n24q02m/Qwen3-Embedding-0.6B-GGUF` | Embedding | 32-1024 (MRL) | 32768 | 378 MB |
| `n24q02m/Qwen3-Reranker-0.6B-GGUF` | Reranker | - | 40960 | 378 MB |

### HuggingFace Repos

| Format | Embedding | Reranker |
|:-------|:----------|:--------|
| ONNX | [n24q02m/Qwen3-Embedding-0.6B-ONNX](https://huggingface.co/n24q02m/Qwen3-Embedding-0.6B-ONNX) | [n24q02m/Qwen3-Reranker-0.6B-ONNX](https://huggingface.co/n24q02m/Qwen3-Reranker-0.6B-ONNX) |
| GGUF | [n24q02m/Qwen3-Embedding-0.6B-GGUF](https://huggingface.co/n24q02m/Qwen3-Embedding-0.6B-GGUF) | [n24q02m/Qwen3-Reranker-0.6B-GGUF](https://huggingface.co/n24q02m/Qwen3-Reranker-0.6B-GGUF) |

## Installation

```bash
pip install qwen3-embed

# For GGUF support
pip install qwen3-embed[gguf]
```

## Usage

### Text Embedding

```python
from qwen3_embed import TextEmbedding

# INT8 (default)
model = TextEmbedding(model_name="n24q02m/Qwen3-Embedding-0.6B-ONNX")

# Q4F16 (smaller, slightly less accurate)
model = TextEmbedding(model_name="n24q02m/Qwen3-Embedding-0.6B-ONNX-Q4F16")

# GGUF (requires: pip install qwen3-embed[gguf])
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
from qwen3_embed import TextCrossEncoder

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
RoPE positions correct regardless of batch composition. See issue
[#725](https://github.com/n24q02m/qwen3-embed/issues/725).

### Custom models (bring your own)

Qwen3 is the only built-in model, but any ONNX-able embedding model can be
registered and then loaded by id. Use `CustomModelSpec` with one of the four
output shapes: `CLS`/`MEAN` (bert-bi), `LAST_TOKEN` (causal), or `DISABLED` (raw).

```python
from qwen3_embed import CustomModelSpec, TextEmbedding

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
from qwen3_embed import CustomRerankerSpec, TextCrossEncoder

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
# pip install "optimum[exporters]" torch transformers onnx
from qwen3_embed.export import export_to_onnx

export_to_onnx("intfloat/multilingual-e5-base", "./e5-onnx")
```

## Configuration

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
from qwen3_embed import TextEmbedding, Device

# Auto-detect GPU (default)
model = TextEmbedding(model_name="n24q02m/Qwen3-Embedding-0.6B-ONNX")

# Force CPU
model = TextEmbedding(model_name="n24q02m/Qwen3-Embedding-0.6B-ONNX", cuda=Device.CPU)

# Force CUDA
model = TextEmbedding(model_name="n24q02m/Qwen3-Embedding-0.6B-ONNX", cuda=Device.CUDA)
```

#### GGUF

GPU is handled by `llama-cpp-python`. The default `pip install qwen3-embed[gguf]` is CPU-only.
For CUDA GPU support, build with:

```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install qwen3-embed[gguf]
```

```python
from qwen3_embed import TextEmbedding, Device

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

## Related Projects

- [wet-mcp](https://github.com/n24q02m/wet-mcp) -- MCP web search server with vector-based docs search, uses qwen3-embed for local embedding
- [mnemo-mcp](https://github.com/n24q02m/mnemo-mcp) -- MCP memory server with semantic search powered by qwen3-embed
- [better-code-review-graph](https://github.com/n24q02m/better-code-review-graph) -- Knowledge graph for code reviews, uses qwen3-embed for local ONNX embedding
- [modalcom-ai-workers](https://github.com/n24q02m/modalcom-ai-workers) -- GPU-serverless workers that convert Qwen3 models to ONNX/GGUF format

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0 -- See [LICENSE](LICENSE). Original fastembed by [Qdrant](https://github.com/qdrant/fastembed).