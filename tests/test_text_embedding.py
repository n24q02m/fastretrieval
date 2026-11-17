from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from fastretrieval.text.qwen3_embedding import Qwen3TextEmbedding
from fastretrieval.text.text_embedding import TextEmbedding
from tests.onnx_runtime_stub import StubSession, install_stub_runtime, write_tokenizer_files


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


_QWEN3_EMBED_MODEL = "n24q02m/Qwen3-Embedding-0.6B-ONNX"


def _hidden_state_from_token_ids(feed: dict[str, np.ndarray]) -> np.ndarray:
    """Every token's hidden state is ``[token_id, 1, 2, 2]``, so pooling is easy to predict."""
    ids = feed["input_ids"].astype(np.float32)
    ones = np.ones_like(ids)
    return np.stack([ids, ones, 2 * ones, 2 * ones], axis=-1)


def _unit(vector: list[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32)
    return array / np.linalg.norm(array)


@pytest.fixture
def stub_text_embedding(tmp_path, monkeypatch):
    """A real TextEmbedding facade over the Qwen3 backend, loaded through a stub runtime."""
    model_dir = tmp_path / "model"
    vocab = write_tokenizer_files(model_dir, ["alpha", "beta"])
    session = StubSession(_hidden_state_from_token_ids)
    # onnxruntime boundary: no CPython 3.15 wheel exists, so the session is a numpy stand-in.
    opened = install_stub_runtime(monkeypatch, session)
    embedding = TextEmbedding(
        _QWEN3_EMBED_MODEL, cache_dir=str(tmp_path / "cache"), specific_model_path=str(model_dir)
    )
    return embedding, session, vocab, model_dir, opened


def test_init_builds_the_registered_qwen3_backend_and_loads_its_graph(stub_text_embedding):
    embedding, _, _, model_dir, opened = stub_text_embedding

    assert type(embedding.model) is Qwen3TextEmbedding
    assert embedding.model.model_description.model == _QWEN3_EMBED_MODEL
    assert opened == [str(model_dir / "onnx" / "model_quantized.onnx")]


def test_query_embed_embeds_a_single_query_string(stub_text_embedding):
    embedding, session, vocab, _, _ = stub_text_embedding

    (vector,) = list(embedding.query_embed("alpha"))

    assert len(session.feeds) == 1
    assert session.feeds[0]["input_ids"][0, -1] == vocab["alpha"]
    np.testing.assert_allclose(vector, _unit([vocab["alpha"], 1, 2, 2]), rtol=1e-6)


def test_query_embed_embeds_queries_streamed_from_a_generator(stub_text_embedding):
    embedding, _, vocab, _, _ = stub_text_embedding

    vectors = list(embedding.query_embed(q for q in ["alpha", "beta"]))

    np.testing.assert_allclose(vectors[0], _unit([vocab["alpha"], 1, 2, 2]), rtol=1e-6)
    np.testing.assert_allclose(vectors[1], _unit([vocab["beta"], 1, 2, 2]), rtol=1e-6)


def test_passage_embed_embeds_passages_streamed_from_a_generator(stub_text_embedding):
    embedding, session, vocab, _, _ = stub_text_embedding

    vectors = list(embedding.passage_embed(t for t in ["beta alpha", "alpha beta"]))

    assert [feed["input_ids"].tolist() for feed in session.feeds] == [
        [[vocab["beta"], vocab["alpha"]]],
        [[vocab["alpha"], vocab["beta"]]],
    ]
    np.testing.assert_allclose(vectors[0], _unit([vocab["alpha"], 1, 2, 2]), rtol=1e-6)
    np.testing.assert_allclose(vectors[1], _unit([vocab["beta"], 1, 2, 2]), rtol=1e-6)
