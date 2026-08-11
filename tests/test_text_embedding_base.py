from collections.abc import Iterable
from typing import Any

import numpy as np
import pytest

from qwen3_embed.common.model_description import DenseModelDescription
from qwen3_embed.common.types import NumpyArray
from qwen3_embed.text.text_embedding_base import TextEmbeddingBase


class MockEmbedding(TextEmbeddingBase):
    """A concrete implementation of TextEmbeddingBase for testing purposes."""

    @classmethod
    def _list_supported_models(cls) -> list[DenseModelDescription]:
        return []

    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
        **kwargs: Any,
    ) -> Iterable[NumpyArray]:
        # Simple mock implementation that yields dummy arrays
        if isinstance(documents, str):
            documents = [documents]
        for _ in documents:
            yield np.zeros(10)

    @classmethod
    def get_embedding_size(cls, model_name: str) -> int:
        return 10

    @property
    def embedding_size(self) -> int:
        return 10

    def token_count(self, texts: str | Iterable[str], **kwargs: Any) -> int:
        return 5


def test_text_embedding_base_init():
    """Test that TextEmbeddingBase initializes its attributes correctly."""
    model_name = "test-model"
    cache_dir = "/tmp/cache"
    threads = 4
    local_files_only = True

    # We instantiate the mock because the base class is abstract (uses NotImplementedError)
    model = MockEmbedding(
        model_name=model_name,
        cache_dir=cache_dir,
        threads=threads,
        local_files_only=local_files_only,
    )

    assert model.model_name == model_name
    assert model.cache_dir == cache_dir
    assert model.threads == threads
    assert model._local_files_only is True
    assert model._embedding_size is None


def test_passage_embed_delegation():
    """Test that passage_embed correctly delegates to the embed method."""
    model = MockEmbedding(model_name="test-model")
    texts = ["text1", "text2"]

    embeddings = list(model.passage_embed(texts))

    assert len(embeddings) == 2
    for emb in embeddings:
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (10,)


def test_query_embed_single_string():
    """Test that query_embed handles a single string correctly."""
    model = MockEmbedding(model_name="test-model")
    query = "test query"

    embeddings = list(model.query_embed(query))

    assert len(embeddings) == 1
    assert isinstance(embeddings[0], np.ndarray)
    assert embeddings[0].shape == (10,)


def test_query_embed_iterable():
    """Test that query_embed handles an iterable of strings correctly."""
    model = MockEmbedding(model_name="test-model")
    queries = ["query1", "query2"]

    embeddings = list(model.query_embed(queries))

    assert len(embeddings) == 2
    for emb in embeddings:
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (10,)


def test_base_class_raises_not_implemented():
    """Test that the base class methods raise NotImplementedError when called directly or via incomplete subclass."""

    class IncompleteEmbedding(TextEmbeddingBase):
        @classmethod
        def _list_supported_models(cls) -> list[DenseModelDescription]:
            return []

    model = IncompleteEmbedding(model_name="test-model")

    with pytest.raises(NotImplementedError):
        list(model.embed("test"))

    with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
        IncompleteEmbedding.get_embedding_size("test-model")

    with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
        _ = model.embedding_size

    with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
        model.token_count("test")
