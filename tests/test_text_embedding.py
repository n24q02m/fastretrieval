from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qwen3_embed.text.text_embedding import TextEmbedding


def test_init_unsupported_model_raises_value_error():
    """Verify that TextEmbedding raises a ValueError when initialized with an unsupported model."""
    with pytest.raises(ValueError, match="is not supported in TextEmbedding"):
        TextEmbedding(model_name="unsupported-model-name")


def test_list_supported_models():
    """Verify that list_supported_models returns a list of dictionaries with model descriptions."""
    models = TextEmbedding.list_supported_models()
    assert isinstance(models, list)
    assert len(models) > 0

    # Check that each item is a dictionary with expected keys
    for model in models:
        assert isinstance(model, dict)
        assert "model" in model
        assert "dim" in model
        assert "description" in model
        assert "size_in_GB" in model
        assert "sources" in model


@pytest.fixture
def mocked_text_embedding():
    """Fixture to provide a TextEmbedding instance with a mocked model."""
    with patch.object(TextEmbedding, "__init__", return_value=None):
        te = TextEmbedding()
        te.model = MagicMock()
        te.model_name = "dummy-model"
        te._embedding_size = None
        return te


def test_passage_embed_delegates_to_model(mocked_text_embedding):
    """Verify that passage_embed correctly delegates to the underlying model."""
    te = mocked_text_embedding
    expected_result = [np.array([1, 2]), np.array([3, 4])]
    te.model.passage_embed.return_value = iter(expected_result)

    texts = ["passage 1", "passage 2"]
    results = list(te.passage_embed(texts, some_arg="value"))

    te.model.passage_embed.assert_called_once()
    args, kwargs = te.model.passage_embed.call_args

    assert list(args[0]) == texts
    assert kwargs == {"some_arg": "value"}
    assert results == expected_result


def test_embed_delegates_to_model(mocked_text_embedding):
    """Verify that embed correctly delegates to the underlying model."""
    te = mocked_text_embedding
    expected_result = [np.array([1, 2])]
    te.model.embed.return_value = iter(expected_result)

    docs = ["doc 1"]
    results = list(te.embed(docs, batch_size=10, parallel=2, other="val"))

    te.model.embed.assert_called_once()
    args, kwargs = te.model.embed.call_args

    assert list(args[0]) == docs
    assert args[1] == 10
    assert args[2] == 2
    assert kwargs == {"other": "val"}
    assert results == expected_result


def test_query_embed_delegates_to_model(mocked_text_embedding):
    """Verify that query_embed correctly delegates to the underlying model."""
    te = mocked_text_embedding
    expected_result = [np.array([5, 6])]
    te.model.query_embed.return_value = iter(expected_result)

    queries = ["query 1"]
    results = list(te.query_embed(queries, extra="foo"))

    te.model.query_embed.assert_called_once()
    args, kwargs = te.model.query_embed.call_args

    assert list(args[0]) == queries
    assert kwargs == {"extra": "foo"}
    assert results == expected_result


def test_token_count_delegates_to_model(mocked_text_embedding):
    """Verify that token_count correctly delegates to the underlying model."""
    te = mocked_text_embedding
    te.model.token_count.return_value = 42

    texts = ["some text to count"]
    count = te.token_count(texts, batch_size=100, custom="opt")

    te.model.token_count.assert_called_once_with(texts, batch_size=100, custom="opt")
    assert count == 42


def test_get_embedding_size():
    """Verify get_embedding_size returns correct dimensions and handles invalid models."""
    models = TextEmbedding.list_supported_models()
    first_model = models[0]
    model_name = first_model["model"]
    expected_dim = first_model["dim"]

    assert TextEmbedding.get_embedding_size(model_name) == expected_dim
    assert TextEmbedding.get_embedding_size(model_name.upper()) == expected_dim

    with pytest.raises(ValueError, match="Embedding size for model unknown-model was None"):
        TextEmbedding.get_embedding_size("unknown-model")


def test_embedding_size_property(mocked_text_embedding):
    """Verify the embedding_size property caches and returns the correct value."""
    te = mocked_text_embedding

    with patch.object(TextEmbedding, "get_embedding_size", return_value=123):
        assert te.embedding_size == 123
        assert te._embedding_size == 123

        TextEmbedding.get_embedding_size.reset_mock()  # type: ignore[attr-defined]
        assert te.embedding_size == 123
        TextEmbedding.get_embedding_size.assert_not_called()  # type: ignore[attr-defined]
