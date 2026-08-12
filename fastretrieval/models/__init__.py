"""Các facade embedding theo modality của fastretrieval."""

from fastretrieval.models.sparse import SparseEmbedding, SparseTextEmbedding

__all__ = ["ImageEmbedding", "SparseEmbedding", "SparseTextEmbedding"]


def __getattr__(name: str) -> object:
    """Nạp model ảnh khi người dùng thực sự yêu cầu capability này."""
    if name == "ImageEmbedding":
        from fastretrieval.models.image import ImageEmbedding

        return ImageEmbedding
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
