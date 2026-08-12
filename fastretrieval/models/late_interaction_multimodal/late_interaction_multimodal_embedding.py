# Chuyển chọn từ qdrant/fastembed 0.8.0 (Apache-2.0); xem NOTICE.
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from typing import Any

from fastretrieval.common.model_description import DenseModelDescription
from fastretrieval.common.types import Device, NumpyArray, OnnxProvider
from fastretrieval.models.image.image_embedding_base import ImageInput
from fastretrieval.models.late_interaction_multimodal.colpali import ColPali
from fastretrieval.models.late_interaction_multimodal.late_interaction_multimodal_embedding_base import (
    LateInteractionMultimodalEmbeddingBase,
)


class LateInteractionMultimodalEmbedding(LateInteractionMultimodalEmbeddingBase):
    EMBEDDINGS_REGISTRY: list[type[LateInteractionMultimodalEmbeddingBase]] = [ColPali]

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
        """Liệt kê các model late-interaction đa modality được hỗ trợ."""
        return [asdict(model) for model in cls._list_supported_models()]

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        result: list[DenseModelDescription] = []
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
        for embedding_model_type in self.EMBEDDINGS_REGISTRY:
            if any(
                model_name.lower() == model.model.lower()
                for model in embedding_model_type._list_supported_models()
            ):
                self.model = embedding_model_type(
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
            f"Model {model_name} is not supported in LateInteractionMultimodalEmbedding. "
            "Please check the supported models using "
            "`LateInteractionMultimodalEmbedding.list_supported_models()`"
        )

    @property
    def embedding_size(self) -> int:
        """Trả về kích thước embedding của model hiện tại."""
        if self._embedding_size is None:
            self._embedding_size = self.get_embedding_size(self.model_name)
        return self._embedding_size

    @classmethod
    def get_embedding_size(cls, model_name: str) -> int:
        """Trả về kích thước embedding của ``model_name``."""
        for description in cls._list_supported_models():
            if description.model.lower() == model_name.lower():
                if description.dim is not None:
                    return description.dim
                break
        model_names = [description.model for description in cls._list_supported_models()]
        raise ValueError(
            f"Embedding size for model {model_name} was None. Available model names: {model_names}"
        )

    def embed_text(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """Mã hóa văn bản trong không gian token-level chung với ảnh."""
        yield from self.model.embed_text(documents, batch_size, parallel, **kwargs)

    def embed_image(
        self,
        images: ImageInput | Iterable[ImageInput],
        batch_size: int = 16,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """Mã hóa ảnh trong không gian token-level chung với văn bản."""
        yield from self.model.embed_image(images, batch_size, parallel, **kwargs)

    def token_count(
        self,
        texts: str | Iterable[str],
        batch_size: int = 1024,
        include_extension: bool = False,
        **kwargs: Any,
    ) -> int:
        """Trả về số token trong truy vấn ColPali."""
        return self.model.token_count(
            texts, batch_size=batch_size, include_extension=include_extension, **kwargs
        )
