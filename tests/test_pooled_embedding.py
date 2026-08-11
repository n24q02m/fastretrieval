"""Unit tests for PooledEmbedding class."""

import numpy as np
import pytest

from qwen3_embed.common.onnx_model import OnnxOutputContext
from qwen3_embed.text.pooled_embedding import PooledEmbedding


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
