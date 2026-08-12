# Chuyển chọn từ qdrant/fastembed 0.8.0 (Apache-2.0); xem NOTICE.
from collections.abc import Iterable
from typing import Any

from fastretrieval.common.model_description import DenseModelDescription
from fastretrieval.common.model_management import ModelManagement
from fastretrieval.common.types import NumpyArray

# Pillow là optional dependency; không import kiểu ảnh ở module base.
ImageInput = Any


class ImageEmbeddingBase(ModelManagement[DenseModelDescription]):
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

    def embed(
        self,
        images: ImageInput | Iterable[ImageInput],
        batch_size: int = 16,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        """Mã hóa ảnh thành vector embedding."""
        raise NotImplementedError()

    @classmethod
    def get_embedding_size(cls, model_name: str) -> int:
        """Trả về kích thước embedding của model đã chọn."""
        raise NotImplementedError("Subclasses must implement this method")

    @property
    def embedding_size(self) -> int:
        """Trả về kích thước embedding của instance hiện tại."""
        raise NotImplementedError("Subclasses must implement this method")
