# Chuyển chọn từ qdrant/fastembed 0.8.0 (Apache-2.0); xem NOTICE.
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from typing import Any

from fastretrieval.common.model_description import SparseModelDescription
from fastretrieval.common.types import Device, OnnxProvider
from fastretrieval.models.sparse.sparse_embedding_base import (
    SparseEmbedding,
    SparseTextEmbeddingBase,
)
from fastretrieval.models.sparse.splade_pp import SpladePP


class SparseTextEmbedding(SparseTextEmbeddingBase):
    EMBEDDINGS_REGISTRY: list[type[SparseTextEmbeddingBase]] = [SpladePP]

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        """List registered sparse model metadata.

        The registry includes a deprecated SPLADE spelling for compatibility;
        prefer the canonical ``prithivida/Splade_PP_en_v1`` model name.

        Returns:
            list[dict[str, Any]]: Model names, artifact sources, and vocabulary sizes.
        """
        return [asdict(model) for model in cls._list_supported_models()]

    @classmethod
    def _list_supported_models(cls) -> list[SparseModelDescription]:
        result: list[SparseModelDescription] = []
        for embedding in cls.EMBEDDINGS_REGISTRY:
            result.extend(embedding._list_supported_models())
        return result

    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        threads: int | None = None,
        providers: Sequence[OnnxProvider] | None = None,
        cuda: bool | Device = Device.AUTO,
        device_ids: list[int] | None = None,
        lazy_load: bool = False,
        **kwargs: Any,
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)
        if model_name.lower() == "prithvida/Splade_PP_en_v1".lower():
            warnings.warn(
                "The right spelling is prithivida/Splade_PP_en_v1. "
                "Support of this name will be removed soon, please fix the model_name",
                DeprecationWarning,
                stacklevel=2,
            )
            model_name = "prithivida/Splade_PP_en_v1"
        for EMBEDDING_MODEL_TYPE in self.EMBEDDINGS_REGISTRY:
            supported_models = EMBEDDING_MODEL_TYPE._list_supported_models()
            if any(model_name.lower() == model.model.lower() for model in supported_models):
                self.model = EMBEDDING_MODEL_TYPE(
                    model_name,
                    cache_dir,
                    threads=threads,
                    providers=providers,
                    cuda=cuda,
                    device_ids=device_ids,
                    lazy_load=lazy_load,
                    **kwargs,
                )
                return
        raise ValueError(
            f"Model {model_name} is not supported in SparseTextEmbedding."
            "Please check the supported models using `SparseTextEmbedding.list_supported_models()`"
        )

    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[SparseEmbedding]:
        """
        Encode a list of documents into list of embeddings.
        We use mean pooling with attention so that the model can handle variable-length inputs.
        Args:
            documents: Iterator of documents or single document to embed
            batch_size: Batch size for encoding -- higher values will use more memory, but be faster
            parallel:
                If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                If 0, use all available cores.
                If None, don't use data-parallel processing, use default onnxruntime threading instead.
        Returns:
            List of embeddings, one per document
        """
        yield from self.model.embed(documents, batch_size, parallel, **kwargs)

    def query_embed(self, query: str | Iterable[str], **kwargs: Any) -> Iterable[SparseEmbedding]:
        """
        Embeds queries
        Args:
            query (Union[str, Iterable[str]]): The query to embed, or an iterable e.g. list of queries.
        Returns:
            Iterable[SparseEmbedding]: The sparse embeddings.
        """
        yield from self.model.query_embed(query, **kwargs)

    def token_count(
        self, texts: str | Iterable[str], batch_size: int = 1024, **kwargs: Any
    ) -> int:
        """Returns the number of tokens in the texts.
        Args:
            texts (str | Iterable[str]): The list of texts to embed.
            batch_size (int): Batch size for encoding
        Returns:
            int: Sum of number of tokens in the texts.
        """
        return self.model.token_count(texts, batch_size=batch_size, **kwargs)
