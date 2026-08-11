"""Qwen3 text embedding via GGUF (llama-cpp-python) with last-token pooling and MRL.

Runtime alternative to ONNX for Qwen3-Embedding. Uses llama-cpp-python for
inference from GGUF quantized models (Q4_K_M).

Requires optional dependency: pip install qwen3-embed[gguf]
"""

import itertools
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from qwen3_embed.common.model_description import DenseModelDescription, ModelSource
from qwen3_embed.common.types import Device, NumpyArray
from qwen3_embed.common.utils import define_cache_dir
from qwen3_embed.text.text_embedding_base import TextEmbeddingBase

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
supported_qwen3_gguf_models: list[DenseModelDescription] = [
    DenseModelDescription(
        model="n24q02m/Qwen3-Embedding-0.6B-GGUF",
        dim=1024,
        description=(
            "Qwen3 text embedding (0.6B) with last-token pooling and MRL support "
            "(32-1024 dims). GGUF Q4_K_M quantized. Multilingual, 32768 input tokens, "
            "instruction-aware, 2025 year."
        ),
        license="apache-2.0",
        size_in_GB=0.48,
        sources=ModelSource(hf="n24q02m/Qwen3-Embedding-0.6B-GGUF"),
        model_file="qwen3-embedding-0.6b-q4-k-m.gguf",
    ),
]

# ---------------------------------------------------------------------------
# Instruction template (same as ONNX version)
# ---------------------------------------------------------------------------
DEFAULT_TASK = "Given a query, retrieve relevant documents that answer the query"
QUERY_INSTRUCTION_TEMPLATE = "Instruct: {task}\nQuery: {text}"


def _check_llama_cpp() -> None:
    """Check that llama-cpp-python is installed."""
    try:
        import llama_cpp  # type: ignore[unresolved-import] # noqa: F401
    except ImportError as e:
        msg = (
            "llama-cpp-python is required for GGUF models. "
            "Install with: pip install qwen3-embed[gguf]"
        )
        raise ImportError(msg) from e


# ---------------------------------------------------------------------------
# GGUF embedding implementation
# ---------------------------------------------------------------------------
class Qwen3TextEmbeddingGGUF(TextEmbeddingBase):
    """Qwen3 GGUF Embedding model with last-token pooling and MRL support.

    Uses llama-cpp-python for inference. Compared to ONNX variants:
    - Smaller model files (Q4_K_M ~480MB vs INT8 ~570MB)
    - Slightly higher latency on CPU

    Usage::

        from qwen3_embed import TextEmbedding

        model = TextEmbedding("n24q02m/Qwen3-Embedding-0.6B-GGUF")
        embeddings = list(model.embed(["Hello world"]))

        # MRL: reduce dimension
        embeddings_256 = list(model.embed(["Hello world"], dim=256))
    """

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        return supported_qwen3_gguf_models

    def __init__(
        self,
        model_name: str = "n24q02m/Qwen3-Embedding-0.6B-GGUF",
        cache_dir: str | None = None,
        threads: int | None = None,
        cuda: bool | Device = Device.AUTO,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the Qwen3 GGUF Embedding model.

        Args:
            model_name: The name of the model to use.
            cache_dir: The path to the cache directory.
            threads: The number of threads to use.
            cuda: Whether to use CUDA for inference.
            **kwargs: Additional arguments, including those for compatibility with the
                TextEmbedding dispatcher (e.g., providers, device_ids, lazy_load).
        """
        _check_llama_cpp()
        super().__init__(model_name, cache_dir, threads, **kwargs)

        self.model_description = self._get_model_description(model_name)
        self.cache_dir = str(define_cache_dir(cache_dir))

        self._model_dir = self.download_model(
            self.model_description,
            self.cache_dir,
            local_files_only=self._local_files_only,
        )

        # Resolve GGUF file
        model_path = Path(self._model_dir) / self.model_description.model_file
        if not model_path.exists():
            msg = f"GGUF model file not found: {model_path}"
            raise FileNotFoundError(msg)

        from llama_cpp import Llama  # type: ignore[unresolved-import]

        # AUTO/-1: offload all layers to GPU if available, fallback to CPU
        # CPU/False/0: force CPU only
        n_gpu = 0 if (cuda is False or cuda == Device.CPU) else -1
        self._llm = Llama(
            model_path=str(model_path),
            embedding=True,
            n_ctx=32768,
            pooling_type=3,  # LLAMA_POOLING_TYPE_LAST
            n_threads=threads or 0,  # 0 = auto-detect
            n_gpu_layers=n_gpu,
            verbose=False,
        )

    # ------------------------------------------------------------------
    # embed / query_embed / passage_embed
    # ------------------------------------------------------------------
    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,  # noqa: ARG002
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """Encode documents into embeddings.

        Args:
            documents: Single document string or iterable of documents.
            batch_size: Number of documents to encode at once.
            parallel: Ignored (single-threaded for GGUF).
            **kwargs: dim (int) enables MRL truncation.

        Yields:
            NumpyArray: L2-normalised embeddings, one per document.
        """
        if isinstance(documents, str):
            documents = [documents]

        dim: int | None = kwargs.get("dim")

        it = iter(documents)
        while batch := tuple(itertools.islice(it, batch_size)):
            # PERFORMANCE: Pass batches to create_embedding to allow llama-cpp-python parallelize processing
            result = self._llm.create_embedding(list(batch))
            for item in result["data"]:
                embedding = np.array(item["embedding"], dtype=np.float32)

                # MRL: optionally truncate to requested dimension
                if dim is not None:
                    embedding = embedding[:dim]

                # L2 normalize
                # Bolt: Fast L2 norm using .dot (~30% faster than linalg.norm)
                norm_sq = embedding.dot(embedding)
                if norm_sq > 0:
                    embedding = embedding / np.sqrt(norm_sq)

                yield embedding

    def query_embed(self, query: str | Iterable[str], **kwargs: Any) -> Iterable[NumpyArray]:
        """Embed queries with instruction prefix.

        Args:
            query: Single query string or iterable of queries.
            **kwargs: task (str) overrides default retrieval instruction.
                dim (int) enables MRL truncation.

        Yields:
            NumpyArray: L2-normalised query embeddings.
        """
        task = kwargs.pop("task", DEFAULT_TASK)
        # ⚡ Bolt: Pre-format constant template parts and use string concatenation (~85% faster)
        prefix, suffix = QUERY_INSTRUCTION_TEMPLATE.format(task=task, text="|||").split("|||")
        if isinstance(query, str):
            queries = [prefix + query + suffix]
        else:
            queries = (prefix + q + suffix for q in query)
        yield from self.embed(queries, **kwargs)

    def passage_embed(self, texts: Iterable[str], **kwargs: Any) -> Iterable[NumpyArray]:
        """Embed passages without instruction prefix.

        Args:
            texts: Iterable of passage strings.
            **kwargs: dim (int) enables MRL truncation.

        Yields:
            NumpyArray: L2-normalised passage embeddings.
        """
        yield from self.embed(texts, **kwargs)
