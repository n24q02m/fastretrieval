from collections.abc import Iterable
from typing import Any

import numpy as np
from numpy.typing import NDArray

from qwen3_embed.common.model_description import (
    CustomDenseModelDescription,
    DenseModelDescription,
    PoolingType,
)
from qwen3_embed.common.onnx_model import OnnxOutputContext
from qwen3_embed.common.types import NumpyArray
from qwen3_embed.common.utils import last_token_pool, mean_pooling, normalize
from qwen3_embed.text.onnx_embedding import OnnxTextEmbedding, OnnxTextEmbeddingWorker


class CustomTextEmbedding(OnnxTextEmbedding):
    # Single source of truth: model-id (lowercased) -> description carrying
    # pooling+normalization. The description is a frozen, picklable dataclass so it
    # survives the deepcopy into spawned worker processes.
    _SUPPORTED: dict[str, CustomDenseModelDescription] = {}

    def __init__(
        self,
        model_name: str,
        **kwargs: Any,
    ):
        super().__init__(
            model_name=model_name,
            **kwargs,
        )
        desc = self._resolve_description(model_name)
        self._pooling = desc.pooling
        self._normalization = desc.normalization

    @classmethod
    def _register(cls, desc: CustomDenseModelDescription) -> None:
        cls._clear_model_cache()
        cls._SUPPORTED[desc.model.lower()] = desc

    @classmethod
    def _resolve_description(cls, model_name: str) -> CustomDenseModelDescription:
        return cls._SUPPORTED[model_name.lower()]

    @classmethod
    def _export_registry(cls) -> list[CustomDenseModelDescription]:
        return list(cls._SUPPORTED.values())

    @classmethod
    def _import_registry(cls, payload: list[CustomDenseModelDescription]) -> None:
        for desc in payload:
            cls._SUPPORTED[desc.model.lower()] = desc

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        return list(cls._SUPPORTED.values())

    @classmethod
    def _get_worker_class(cls) -> type["CustomTextEmbeddingWorker"]:
        return CustomTextEmbeddingWorker

    def _extra_worker_params(self) -> dict[str, Any]:
        # Propagate the runtime registry so spawned workers (fresh interpreters
        # with an empty _SUPPORTED) can resolve + re-register this custom model.
        return {"custom_registry": self._export_registry()}

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        embeddings = self._pool(output.model_output, output.attention_mask)

        # MRL: optionally truncate to requested dimension
        dim: int | None = kwargs.get("dim")
        if dim is not None:
            embeddings = embeddings[:, :dim]

        return self._normalize(embeddings)

    def _pool(
        self, embeddings: NumpyArray, attention_mask: NDArray[np.int64] | None = None
    ) -> NumpyArray:
        if self._pooling == PoolingType.CLS:
            return embeddings[:, 0]

        if self._pooling == PoolingType.MEAN:
            if attention_mask is None:
                raise ValueError("attention_mask must be provided for mean pooling")
            return mean_pooling(embeddings, attention_mask)

        if self._pooling == PoolingType.LAST_TOKEN:
            if attention_mask is None:
                raise ValueError("attention_mask must be provided for last-token pooling")
            return last_token_pool(embeddings, attention_mask)

        if self._pooling == PoolingType.DISABLED:
            return embeddings

        raise ValueError(
            f"Unsupported pooling type {self._pooling}. "
            f"Supported types are: {PoolingType.CLS}, {PoolingType.MEAN}, "
            f"{PoolingType.LAST_TOKEN}, {PoolingType.DISABLED}."
        )

    def _normalize(self, embeddings: NumpyArray) -> NumpyArray:
        return normalize(embeddings) if self._normalization else embeddings


class CustomTextEmbeddingWorker(OnnxTextEmbeddingWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> CustomTextEmbedding:
        registry = kwargs.pop("custom_registry", None)
        if registry is not None:
            CustomTextEmbedding._import_registry(registry)
        return CustomTextEmbedding(
            model_name=model_name,
            cache_dir=cache_dir,
            threads=1,
            **kwargs,
        )
