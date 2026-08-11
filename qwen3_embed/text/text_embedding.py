from collections.abc import Iterable, Sequence
from dataclasses import asdict
from typing import Any

from qwen3_embed.common.model_description import (
    CustomDenseModelDescription,
    DenseModelDescription,
    PoolingType,
)
from qwen3_embed.common.types import Device, NumpyArray, OnnxProvider
from qwen3_embed.text.custom_text_embedding import CustomTextEmbedding
from qwen3_embed.text.gguf_embedding import Qwen3TextEmbeddingGGUF
from qwen3_embed.text.onnx_embedding import OnnxTextEmbedding
from qwen3_embed.text.pooled_embedding import PooledEmbedding
from qwen3_embed.text.pooled_normalized_embedding import PooledNormalizedEmbedding
from qwen3_embed.text.qwen3_embedding import Qwen3TextEmbedding
from qwen3_embed.text.text_embedding_base import TextEmbeddingBase


class TextEmbedding(TextEmbeddingBase):
    EMBEDDINGS_REGISTRY: list[type[TextEmbeddingBase]] = [
        Qwen3TextEmbedding,
        Qwen3TextEmbeddingGGUF,
        OnnxTextEmbedding,
        PooledNormalizedEmbedding,
        PooledEmbedding,
        CustomTextEmbedding,
    ]
    _supported_models_cache: list[DenseModelDescription] | None = None
    _embedding_type_cache: dict[str, type[TextEmbeddingBase]] | None = None
    _embedding_description_cache: dict[str, DenseModelDescription] | None = None

    @classmethod
    def _clear_model_cache(cls) -> None:
        cls._embedding_type_cache = None
        cls._supported_models_cache = None
        cls._embedding_description_cache = None

    @classmethod
    def _build_caches(cls) -> None:
        if cls._embedding_type_cache is not None and cls._embedding_description_cache is not None:
            return

        new_type_cache = {}
        new_desc_cache = {}

        for EMBEDDING_MODEL_TYPE in cls.EMBEDDINGS_REGISTRY:
            for model in EMBEDDING_MODEL_TYPE._list_supported_models():
                model_lower = model.model.lower()
                new_type_cache[model_lower] = EMBEDDING_MODEL_TYPE
                new_desc_cache[model_lower] = model

        cls._embedding_type_cache = new_type_cache
        cls._embedding_description_cache = new_desc_cache

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        """Lists the supported models.

        Returns:
            list[dict[str, Any]]: A list of dictionaries containing the model information.
        """
        return [asdict(model) for model in cls._list_supported_models()]

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        if cls._supported_models_cache is not None:
            return cls._supported_models_cache

        result: list[DenseModelDescription] = []
        for embedding in cls.EMBEDDINGS_REGISTRY:
            result.extend(embedding._list_supported_models())

        cls._supported_models_cache = result
        return result

    @classmethod
    def add_custom_model(
        cls,
        model_description: DenseModelDescription,
        pooling: PoolingType,
        normalization: bool,
    ) -> None:
        cls._build_caches()
        assert cls._embedding_type_cache is not None

        model_lower = model_description.model.lower()
        if model_lower in cls._embedding_type_cache:
            raise ValueError(
                f"Model {model_description.model} is already registered in TextEmbedding, if you still want to add this model, "
                f"please use another model name"
            )

        cls._clear_model_cache()
        carrying = CustomDenseModelDescription(
            model=model_description.model,
            dim=model_description.dim,
            sources=model_description.sources,
            model_file=model_description.model_file,
            description=model_description.description,
            license=model_description.license,
            size_in_GB=model_description.size_in_GB,
            additional_files=model_description.additional_files,
            pooling=pooling,
            normalization=normalization,
        )
        CustomTextEmbedding._register(carrying)

    def __init__(
        self,
        model_name: str = "n24q02m/Qwen3-Embedding-0.6B-ONNX",
        cache_dir: str | None = None,
        threads: int | None = None,
        providers: Sequence[OnnxProvider] | None = None,
        cuda: bool | Device = Device.AUTO,
        device_ids: list[int] | None = None,
        lazy_load: bool = False,
        **kwargs: Any,
    ):
        super().__init__(model_name, cache_dir, threads, **kwargs)
        self._build_caches()
        assert self._embedding_type_cache is not None

        model_name_lower = model_name.lower()
        EMBEDDING_MODEL_TYPE = self._embedding_type_cache.get(model_name_lower)

        if EMBEDDING_MODEL_TYPE is not None:
            self.model = EMBEDDING_MODEL_TYPE(
                model_name=model_name,
                cache_dir=cache_dir,
                threads=threads,
                providers=providers,
                cuda=cuda,
                device_ids=device_ids,
                lazy_load=lazy_load,
                **kwargs,
            )
            return

        raise ValueError(
            f"Model {model_name} is not supported in TextEmbedding. "
            "Please check the supported models using `TextEmbedding.list_supported_models()`"
        )

    @property
    def embedding_size(self) -> int:
        """Get the embedding size of the current model"""
        if self._embedding_size is None:
            self._embedding_size = self.get_embedding_size(self.model_name)
        return self._embedding_size

    @classmethod
    def get_embedding_size(cls, model_name: str) -> int:
        """Get the embedding size of the passed model

        Args:
            model_name (str): The name of the model to get embedding size for.

        Returns:
            int: The size of the embedding.

        Raises:
            ValueError: If the model name is not found in the supported models.
        """
        cls._build_caches()
        assert cls._embedding_description_cache is not None

        model_name_lower = model_name.lower()
        desc = cls._embedding_description_cache.get(model_name_lower)

        if desc is not None:
            assert desc.dim is not None
            return desc.dim

        model_names = [desc.model for desc in cls._list_supported_models()]
        raise ValueError(
            f"Embedding size for model {model_name} was None. Available model names: {model_names}"
        )

    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """
        Encode a list of documents into list of embeddings.
        Pooling is model-specific (last-token for Qwen3, configurable for custom models),
        so the model can handle variable-length inputs.

        Args:
            documents: Iterator of documents or single document to embed
            batch_size: Batch size for encoding -- higher values will use more memory, but be faster.
                Capped at the batch size the ONNX graph is pinned to, if it declares one.
            parallel:
                If > 1, data-parallel encoding will be used, recommended for offline encoding of large datasets.
                If 0, use all available cores.
                If None, don't use data-parallel processing, use default onnxruntime threading instead.

        Returns:
            List of embeddings, one per document
        """
        from qwen3_embed.common.utils import check_input_length, iter_checked_texts

        if isinstance(documents, str):
            check_input_length(documents)
            docs: str | Iterable[str] = documents
        else:
            docs = iter_checked_texts(documents)

        yield from self.model.embed(docs, batch_size, parallel, **kwargs)

    def query_embed(self, query: str | Iterable[str], **kwargs: Any) -> Iterable[NumpyArray]:
        """
        Embeds queries

        Args:
            query (Union[str, Iterable[str]]): The query to embed, or an iterable e.g. list of queries.

        Returns:
            Iterable[NumpyArray]: The embeddings.
        """
        from qwen3_embed.common.utils import check_input_length, iter_checked_texts

        if isinstance(query, str):
            check_input_length(query)
            q: str | Iterable[str] = query
        else:
            q = iter_checked_texts(query)

        # This is model-specific, so that different models can have specialized implementations
        yield from self.model.query_embed(q, **kwargs)

    def passage_embed(self, texts: Iterable[str], **kwargs: Any) -> Iterable[NumpyArray]:
        """
        Embeds a list of text passages into a list of embeddings.

        Args:
            texts (Iterable[str]): The list of texts to embed.
            **kwargs: Additional keyword argument to pass to the embed method.

        Yields:
            Iterable[NumpyArray]: The passage embeddings, one per text.
        """
        # This is model-specific, so that different models can have specialized implementations
        from qwen3_embed.common.utils import iter_checked_texts

        yield from self.model.passage_embed(iter_checked_texts(texts), **kwargs)

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
