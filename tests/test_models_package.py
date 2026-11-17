"""Behavior of the ``fastretrieval.models`` package namespace."""

import pytest

import fastretrieval.models as models
from fastretrieval.models.image.image_embedding import ImageEmbedding
from fastretrieval.models.sparse import SparseEmbedding, SparseTextEmbedding


def test_models_package_exports_the_sparse_facades_and_lists_image_embedding() -> None:
    assert models.__all__ == ["ImageEmbedding", "SparseEmbedding", "SparseTextEmbedding"]
    assert models.SparseEmbedding is SparseEmbedding
    assert models.SparseTextEmbedding is SparseTextEmbedding


def test_models_package_resolves_image_embedding_on_attribute_access_without_binding_it() -> None:
    from fastretrieval.models import ImageEmbedding as imported

    assert models.ImageEmbedding is ImageEmbedding
    assert imported is ImageEmbedding
    assert "ImageEmbedding" not in vars(models)


def test_models_package_rejects_unknown_attributes_with_attribute_error() -> None:
    with pytest.raises(
        AttributeError, match=r"^module 'fastretrieval\.models' has no attribute 'DoesNotExist'$"
    ):
        _ = models.DoesNotExist
    assert not hasattr(models, "DoesNotExist")
