"""Unit tests for last_token_pool and other utility functions."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from qwen3_embed.common.utils import (
    check_input_length,
    define_cache_dir,
    iter_batch,
    iter_checked_texts,
    last_token_pool,
    mean_pooling,
    normalize,
)


class TestLastTokenPool:
    def test_last_token_pool_right_padding(self):
        input_array = np.array(
            [[[1, 2, 3], [4, 5, 6], [0, 0, 0]], [[7, 8, 9], [0, 0, 0], [0, 0, 0]]],
            dtype=np.float32,
        )
        attention_mask = np.array([[1, 1, 0], [1, 0, 0]], dtype=np.int64)
        expected_output = np.array([[4, 5, 6], [7, 8, 9]], dtype=np.float32)
        output = last_token_pool(input_array, attention_mask)
        assert np.allclose(output, expected_output)

    def test_last_token_pool_left_padding(self):
        input_array = np.array(
            [[[0, 0, 0], [1, 2, 3], [4, 5, 6]], [[0, 0, 0], [0, 0, 0], [7, 8, 9]]],
            dtype=np.float32,
        )
        attention_mask = np.array([[0, 1, 1], [0, 0, 1]], dtype=np.int64)
        expected_output = np.array([[4, 5, 6], [7, 8, 9]], dtype=np.float32)
        output = last_token_pool(input_array, attention_mask)
        assert np.allclose(output, expected_output)


class TestNormalize:
    def test_normalize_l2(self):
        input_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        output = normalize(input_array, p=2, dim=1)
        expected_output = input_array / np.linalg.norm(input_array, ord=2, axis=1, keepdims=True)
        assert np.allclose(output, expected_output)

    def test_normalize_l1(self):
        input_array = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
        output = normalize(input_array, p=1, dim=1)
        expected_output = input_array / np.linalg.norm(input_array, ord=1, axis=1, keepdims=True)
        assert np.allclose(output, expected_output)


class TestMeanPooling:
    def test_mean_pooling(self):
        input_array = np.array(
            [[[1, 2, 3], [4, 5, 6], [0, 0, 0]], [[7, 8, 9], [10, 11, 12], [0, 0, 0]]],
            dtype=np.float32,
        )
        attention_mask = np.array([[1, 1, 0], [1, 1, 0]], dtype=np.int64)
        expected_output = np.array(
            [[(1 + 4) / 2, (2 + 5) / 2, (3 + 6) / 2], [(7 + 10) / 2, (8 + 11) / 2, (9 + 12) / 2]],
            dtype=np.float32,
        )
        output = mean_pooling(input_array, attention_mask)
        assert np.allclose(output, expected_output)


class TestIterBatch:
    def test_list_input(self) -> None:
        """Test with list input."""
        data = [1, 2, 3, 4, 5]
        size = 2
        expected = [[1, 2], [3, 4], [5]]
        assert list(iter_batch(data, size)) == expected

    def test_tuple_input(self) -> None:
        """Test with tuple input."""
        data = (1, 2, 3, 4, 5)
        size = 3
        expected = [[1, 2, 3], [4, 5]]
        assert list(iter_batch(data, size)) == expected

    def test_iterator_input(self) -> None:
        """Test with iterator input."""
        data = iter([1, 2, 3, 4, 5])
        size = 2
        expected = [[1, 2], [3, 4], [5]]
        assert list(iter_batch(data, size)) == expected

    def test_exact_batch_size(self) -> None:
        """Test when data length is exactly divisible by batch size."""
        data = [1, 2, 3, 4]
        size = 2
        expected = [[1, 2], [3, 4]]
        assert list(iter_batch(data, size)) == expected

    def test_zero_batch_size(self) -> None:
        """Test with zero batch size (should return empty list)."""
        data = [1, 2, 3]
        result = list(iter_batch(data, 0))
        assert result == []

    def test_negative_batch_size(self) -> None:
        """Test with negative batch size (should raise ValueError)."""
        data = [1, 2, 3]
        with pytest.raises(ValueError, match="Stop argument for islice"):
            list(iter_batch(data, -1))

    def test_too_large_batch_size(self) -> None:
        """Test with batch size larger than sys.maxsize (should raise ValueError)."""
        data = [1, 2, 3]
        with pytest.raises(ValueError, match="Stop argument for islice"):
            list(iter_batch(data, sys.maxsize + 1))

    def test_empty_iterable(self) -> None:
        """Test with empty iterable."""
        assert list(iter_batch([], 2)) == []
        assert list(iter_batch((), 2)) == []
        assert list(iter_batch(iter([]), 2)) == []

    def test_zero_batch_size_iterator_not_consumed(self) -> None:
        """Test that size=0 does not consume the iterator."""
        data = iter([1, 2, 3])
        result = list(iter_batch(data, 0))
        assert result == []
        assert next(data) == 1


class TestDefineCacheDir:
    @patch("qwen3_embed.common.utils.Path.mkdir")
    @patch("qwen3_embed.common.utils.Path.chmod")
    def test_custom_argument(self, mock_chmod, mock_mkdir):
        res = define_cache_dir("/custom/arg/path")
        assert res == Path("/custom/arg/path")
        mock_mkdir.assert_called_once_with(mode=0o700, parents=True, exist_ok=True)
        mock_chmod.assert_called_once_with(0o700)

    @patch("qwen3_embed.common.utils.Path.mkdir")
    @patch("qwen3_embed.common.utils.Path.chmod")
    @patch.dict(os.environ, {"QWEN3_EMBED_CACHE_PATH": "/custom/env/path"})
    def test_qwen3_cache_path(self, mock_chmod, mock_mkdir):
        res = define_cache_dir()
        assert res == Path("/custom/env/path")
        mock_mkdir.assert_called_once_with(mode=0o700, parents=True, exist_ok=True)
        mock_chmod.assert_called_once_with(0o700)

    @patch("qwen3_embed.common.utils.Path.mkdir")
    @patch("qwen3_embed.common.utils.Path.chmod")
    @patch.dict(os.environ, {"XDG_CACHE_HOME": "/xdg/cache"}, clear=True)
    def test_xdg_cache_home(self, mock_chmod, mock_mkdir):
        res = define_cache_dir()
        assert res == Path("/xdg/cache/qwen3_embed")
        mock_mkdir.assert_called_once_with(mode=0o700, parents=True, exist_ok=True)
        mock_chmod.assert_called_once_with(0o700)

    @patch("qwen3_embed.common.utils.Path.mkdir")
    @patch("qwen3_embed.common.utils.Path.chmod")
    def test_fallback_home_cache(self, mock_chmod, mock_mkdir):
        # Resolve the expected home dir BEFORE clearing the environment, since
        # Path.home() on Windows depends on env vars (USERPROFILE etc.) that
        # clear=True would wipe and cause a RuntimeError.
        fake_home = Path.home()
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("qwen3_embed.common.utils.Path.home", return_value=fake_home),
        ):
            res = define_cache_dir()
        assert res == fake_home / ".cache/qwen3_embed"
        mock_mkdir.assert_called_once_with(mode=0o700, parents=True, exist_ok=True)
        mock_chmod.assert_called_once_with(0o700)

    @patch("qwen3_embed.common.utils.Path.mkdir")
    @patch("qwen3_embed.common.utils.Path.chmod", side_effect=OSError("chmod failed"))
    def test_chmod_oserror_suppressed(self, mock_chmod, mock_mkdir):
        res = define_cache_dir("/custom/arg/path")
        assert res == Path("/custom/arg/path")
        mock_mkdir.assert_called_once_with(mode=0o700, parents=True, exist_ok=True)
        mock_chmod.assert_called_once_with(0o700)


class TestInputValidation:
    """Tests for input length validation utilities."""

    def test_check_input_length_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should not raise error when within limits."""
        import qwen3_embed.common.utils

        monkeypatch.setattr(qwen3_embed.common.utils, "MAX_INPUT_LENGTH", 5)

        # Exact limit
        check_input_length("abcde")
        # Below limit
        check_input_length("abcd")

    def test_check_input_length_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise ValueError when limit is exceeded."""
        import qwen3_embed.common.utils

        monkeypatch.setattr(qwen3_embed.common.utils, "MAX_INPUT_LENGTH", 5)

        with pytest.raises(
            ValueError, match="Input string exceeds maximum allowed length of 5 characters"
        ):
            check_input_length("abcdef")

    def test_iter_checked_texts_valid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Generator should yield all texts if they are within limits."""
        import qwen3_embed.common.utils

        monkeypatch.setattr(qwen3_embed.common.utils, "MAX_INPUT_LENGTH", 5)

        texts = ["abc", "de", "fghij"]
        result = list(iter_checked_texts(texts))
        assert result == texts

    def test_iter_checked_texts_invalid(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Generator should yield valid texts before raising ValueError on invalid one."""
        import qwen3_embed.common.utils

        monkeypatch.setattr(qwen3_embed.common.utils, "MAX_INPUT_LENGTH", 5)

        texts = ["abc", "abcdef", "valid"]
        iterator = iter(iter_checked_texts(texts))

        assert next(iterator) == "abc"
        with pytest.raises(
            ValueError, match="Input string exceeds maximum allowed length of 5 characters"
        ):
            next(iterator)

    def test_check_input_length_extremely_long(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Should raise ValueError when input is extremely long (exceeds 1,000,000)."""
        import qwen3_embed.common.utils

        # Explicitly set to 1,000,000 to avoid interference from other tests
        monkeypatch.setattr(qwen3_embed.common.utils, "MAX_INPUT_LENGTH", 1000000)

        # Test with the default 1,000,000 limit
        with pytest.raises(
            ValueError, match="Input string exceeds maximum allowed length of 1000000 characters"
        ):
            check_input_length("a" * 1000001)

    def test_check_input_length_empty_is_accepted(self) -> None:
        """Empty input is valid: the guard bounds size only, and "" is the cheapest input."""
        check_input_length("")

    def test_check_input_length_whitespace_is_accepted(self) -> None:
        """Whitespace-only input is valid; blank machine-generated text is a real input."""
        check_input_length("   ")
        check_input_length("\n\t ")

    def test_iter_checked_texts_passes_blank_entries_through(self) -> None:
        """A blank entry mid-iterable must not abort the batch."""
        texts = ["abc", "", "   ", "def"]
        assert list(iter_checked_texts(texts)) == texts
