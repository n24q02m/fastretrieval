# Chuyển chọn từ qdrant/fastembed 0.8.0 (Apache-2.0); xem NOTICE.
from collections.abc import Iterable, Sequence
from dataclasses import asdict
from typing import Any

from fastretrieval.common.model_description import DenseModelDescription
from fastretrieval.common.types import Device, NumpyArray, OnnxProvider
from fastretrieval.models.image.image_embedding_base import ImageEmbeddingBase, ImageInput
from fastretrieval.models.image.onnx_embedding import OnnxImageEmbedding


class ImageEmbedding(ImageEmbeddingBase):
    """Facade chọn implementation ONNX theo model registry."""

    EMBEDDINGS_REGISTRY: list[type[ImageEmbeddingBase]] = [OnnxImageEmbedding]

    @classmethod
    def list_supported_models(cls) -> list[dict[str, Any]]:
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
            supported_models = embedding_model_type._list_supported_models()
            if any(model_name.lower() == model.model.lower() for model in supported_models):
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
            f"Model {model_name} is not supported in ImageEmbedding. "
            "Please check ImageEmbedding.list_supported_models()."
        )

    @property
    def embedding_size(self) -> int:
        if self._embedding_size is None:
            self._embedding_size = self.get_embedding_size(self.model_name)
        return self._embedding_size

    @classmethod
    def get_embedding_size(cls, model_name: str) -> int:
        for description in cls._list_supported_models():
            if description.model.lower() == model_name.lower():
                assert description.dim is not None
                return description.dim
        model_names = [description.model for description in cls._list_supported_models()]
        raise ValueError(
            f"Embedding size for model {model_name} is unknown. Available model names: {model_names}"
        )

    def embed(
        self,
        images: ImageInput | Iterable[ImageInput],
        batch_size: int = 16,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        yield from self.model.embed(images, batch_size, parallel, **kwargs)
