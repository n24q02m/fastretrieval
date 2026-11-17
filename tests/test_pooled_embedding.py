"""Unit tests for PooledEmbedding class."""

import numpy as np
import pytest

import fastretrieval.text.pooled_embedding as pooled_embedding_module
from fastretrieval.common.model_description import DenseModelDescription, ModelSource
from fastretrieval.common.onnx_model import OnnxOutputContext
from fastretrieval.text.pooled_embedding import PooledEmbedding, PooledEmbeddingWorker
from tests.onnx_runtime_stub import StubSession, install_stub_runtime, write_tokenizer_files


class MockPooledEmbedding(PooledEmbedding):
    """Mock class to bypass model loading."""

    def __init__(self):
        pass


def test_post_process_raises_without_attention_mask():
    """Test that _post_process_onnx_output raises ValueError if attention_mask is missing."""
    embedding = MockPooledEmbedding()
    output = OnnxOutputContext(
        model_output=np.zeros((1, 3, 4), dtype=np.float32), attention_mask=None
    )
    with pytest.raises(
        ValueError, match="attention_mask must be provided for document post-processing"
    ):
        embedding._post_process_onnx_output(output)


def test_post_process_mean_pooling_works():
    """Test that _post_process_onnx_output correctly performs mean pooling."""
    embedding = MockPooledEmbedding()
    # Batch 1, Seq 2, Dim 2
    model_output = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    attention_mask = np.array([[1, 1]], dtype=np.int64)
    output = OnnxOutputContext(model_output=model_output, attention_mask=attention_mask)

    result = list(embedding._post_process_onnx_output(output))
    # Mean of [[1, 2], [3, 4]] is [2, 3]
    expected = np.array([[2.0, 3.0]], dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_mean_pooling_happy_path():
    """Test standard mean pooling with a typical 3D array and matching mask."""
    model_output = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # Sequence 0
            [[7.0, 8.0], [9.0, 10.0], [11.0, 12.0]],  # Sequence 1
        ],
        dtype=np.float32,
    )
    attention_mask = np.array(
        [
            [1, 1, 0],  # Use first two tokens
            [0, 1, 1],  # Use last two tokens
        ],
        dtype=np.int64,
    )

    result = PooledEmbedding.mean_pooling(model_output, attention_mask)

    # Sequence 0 mean: ([1.0, 2.0] + [3.0, 4.0]) / 2 = [2.0, 3.0]
    # Sequence 1 mean: ([9.0, 10.0] + [11.0, 12.0]) / 2 = [10.0, 11.0]
    expected = np.array([[2.0, 3.0], [10.0, 11.0]], dtype=np.float32)

    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_mean_pooling_empty_mask():
    """Test mean pooling with an all-zero mask to verify division-by-zero avoidance."""
    model_output = np.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=np.float32)
    attention_mask = np.array([[0, 0]], dtype=np.int64)

    result = PooledEmbedding.mean_pooling(model_output, attention_mask)

    # Without division by zero protection, this would produce NaNs or infs.
    # The current implementation divides by max(0, 1e-9), so 0 / 1e-9 = 0.
    expected = np.array([[0.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(result, expected, atol=1e-6)


def test_mean_pooling_varying_lengths():
    """Test a batch where sequences have different numbers of valid tokens."""
    model_output = np.array(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # Length 3 valid
            [[7.0, 8.0], [0.0, 0.0], [0.0, 0.0]],  # Length 1 valid
        ],
        dtype=np.float32,
    )
    attention_mask = np.array(
        [
            [1, 1, 1],
            [1, 0, 0],
        ],
        dtype=np.int64,
    )

    result = PooledEmbedding.mean_pooling(model_output, attention_mask)

    # Sequence 0 mean: ([1, 2] + [3, 4] + [5, 6]) / 3 = [3.0, 4.0]
    # Sequence 1 mean: [7.0, 8.0] / 1 = [7.0, 8.0]
    expected = np.array([[3.0, 4.0], [7.0, 8.0]], dtype=np.float32)

    np.testing.assert_allclose(result, expected, atol=1e-6)


_MEAN_MODEL = "org/mean-pooled"


def _hidden_state_from_token_ids(feed: dict[str, np.ndarray]) -> np.ndarray:
    """Every token's hidden state is ``[token_id, 1, 2, 2]``, so pooling is easy to predict."""
    ids = feed["input_ids"].astype(np.float32)
    ones = np.ones_like(ids)
    return np.stack([ids, ones, 2 * ones, 2 * ones], axis=-1)


@pytest.fixture
def mean_pooled_registry(monkeypatch):
    """Swap in a private registry; ``_list_supported_models`` reads the module list per call."""
    desc = DenseModelDescription(
        model=_MEAN_MODEL,
        dim=4,
        sources=ModelSource(hf=_MEAN_MODEL),
        model_file="onnx/model.onnx",
        description="",
        license="",
        size_in_GB=0.0,
    )
    monkeypatch.setattr(pooled_embedding_module, "supported_pooled_models", [desc])
    PooledEmbedding._clear_model_cache()
    yield desc
    PooledEmbedding._clear_model_cache()


def test_post_process_truncates_mean_pooled_vectors_to_the_requested_dimension():
    embeddings = np.array([[[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0]]], dtype=np.float32)
    mask = np.array([[1, 1]], dtype=np.int64)
    output = OnnxOutputContext(model_output=embeddings, attention_mask=mask)

    pooled = np.asarray(MockPooledEmbedding()._post_process_onnx_output(output, dim=2))

    np.testing.assert_allclose(pooled, [[2.0, 3.0]])


def test_worker_builds_a_single_threaded_mean_pooling_model(
    mean_pooled_registry, tmp_path, monkeypatch
):
    model_dir = tmp_path / "model"
    vocab = write_tokenizer_files(model_dir, ["alpha", "beta"])
    # onnxruntime boundary: no CPython 3.15 wheel exists, so the session is a numpy stand-in.
    install_stub_runtime(monkeypatch, StubSession(_hidden_state_from_token_ids))

    worker = PooledEmbeddingWorker(
        _MEAN_MODEL, str(tmp_path / "cache"), specific_model_path=str(model_dir)
    )
    ((idx, output),) = list(worker.process([(2, ["alpha beta"])]))
    pooled = np.asarray(worker.model._post_process_onnx_output(output))

    assert PooledEmbedding._get_worker_class() is PooledEmbeddingWorker
    assert type(worker.model) is PooledEmbedding
    assert worker.model.threads == 1
    assert idx == 2
    np.testing.assert_allclose(pooled, [[(vocab["alpha"] + vocab["beta"]) / 2, 1.0, 2.0, 2.0]])
