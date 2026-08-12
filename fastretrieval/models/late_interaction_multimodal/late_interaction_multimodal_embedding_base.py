# Chuyển chọn từ qdrant/fastembed 0.8.0 (Apache-2.0); xem NOTICE.
from collections.abc import Iterable
from typing import Any

from fastretrieval.common.model_description import DenseModelDescription
from fastretrieval.common.model_management import ModelManagement
from fastretrieval.common.types import NumpyArray
from fastretrieval.models.image.image_embedding_base import ImageInput


class LateInteractionMultimodalEmbeddingBase(ModelManagement[DenseModelDescription]):
    def __init__(
        self,
        model_name: str,
        cache_dir: str | None = None,
        threads: int | None = None,
        **kwargs: Any,
    ):
        self.model_name = model_name
        self.cache_dir = cache_dir
        self.threads = threads
        self._local_files_only = kwargs.pop("local_files_only", False)
        self._embedding_size: int | None = None

    def embed_text(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """Mã hóa văn bản thành embedding token-level."""
        raise NotImplementedError()

    def embed_image(
        self,
        images: ImageInput | Iterable[ImageInput],
        batch_size: int = 16,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """Mã hóa ảnh thành embedding token-level."""
        raise NotImplementedError()

    @classmethod
    def get_embedding_size(cls, model_name: str) -> int:
        """Trả về kích thước embedding của model đã chọn."""
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def embedding_size(self) -> int:
        """Trả về kích thước embedding của instance hiện tại."""
        raise NotImplementedError("Subclasses must implement this method")

    def token_count(self, texts: str | Iterable[str], **kwargs: Any) -> int:
        """Trả về số token trong văn bản."""
        raise NotImplementedError("Subclasses must implement this method")
