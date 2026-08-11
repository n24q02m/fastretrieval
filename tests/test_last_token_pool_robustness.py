import numpy as np
import pytest

from qwen3_embed.common.utils import last_token_pool


def test_mixed_padding():
    # Row 0: Left padded [0, 1, 1] -> Last token at index 2 (val 3)
    # Row 1: Right padded [1, 1, 0] -> Last token at index 1 (val 5)
    hidden = np.array([[[1, 1], [2, 2], [3, 3]], [[4, 4], [5, 5], [6, 6]]], dtype=np.float32)
    mask = np.array([[0, 1, 1], [1, 1, 0]], dtype=np.int64)

    res = last_token_pool(hidden, mask)
    expected = np.array([[3, 3], [5, 5]], dtype=np.float32)
    np.testing.assert_array_equal(res, expected)


def test_empty_sequence():
    # Case: seq_len = 0
    hidden = np.zeros((2, 0, 4), dtype=np.float32)
    mask = np.zeros((2, 0), dtype=np.int64)

    # This should not crash and should return zeros
    res = last_token_pool(hidden, mask)
    assert res.shape == (2, 4)
    assert np.all(res == 0)


def test_all_zero_mask():
    # Row 0: Normal
    # Row 1: All zeros -> should return zeros
    hidden = np.array([[[1, 1], [2, 2]], [[3, 3], [4, 4]]], dtype=np.float32)
    mask = np.array([[1, 0], [0, 0]], dtype=np.int64)

    res = last_token_pool(hidden, mask)
    # Row 0: last valid is index 0 -> [1, 1]
    # Row 1: all zeros -> returns zeros
    np.testing.assert_allclose(res[0], [1, 1])
    np.testing.assert_allclose(res[1], [0, 0])


def test_discontiguous_mask():
    hidden = np.array([[[1, 1], [2, 2], [3, 3]]], dtype=np.float32)
    mask = np.array([[1, 0, 1]], dtype=np.int64)
    res = last_token_pool(hidden, mask)
    # Last valid is index 2
    np.testing.assert_array_equal(res, [[3, 3]])


def test_2d_input():
    # batch, seq
    hidden = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    mask = np.array([[1, 1, 0], [0, 1, 1]], dtype=np.int64)
    res = last_token_pool(hidden, mask)
    # Row 0: index 1 -> 2
    # Row 1: index 2 -> 6
    np.testing.assert_array_equal(res, [2, 6])


if __name__ == "__main__":
    pytest.main([__file__])
