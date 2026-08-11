from unittest.mock import MagicMock, patch

import pytest

from qwen3_embed.text.text_embedding import TextEmbedding


class TestTextEmbeddingEmbed:
    @pytest.fixture
    def mock_embedding(self):
        with patch("qwen3_embed.text.text_embedding.TextEmbedding.__init__", return_value=None):
            embedding = TextEmbedding("dummy-model")
            embedding.model = MagicMock()
            return embedding

    def test_embed_single_string(self, mock_embedding):
        """Verify embed calls check_input_length and delegates to model.embed for a single string."""
        mock_embedding.model.embed.return_value = iter([1, 2, 3])
        doc = "test document"

        with patch("qwen3_embed.common.utils.check_input_length") as mock_check:
            results = list(mock_embedding.embed(doc, batch_size=128, parallel=1, extra="param"))

            mock_check.assert_called_once_with(doc)
            mock_embedding.model.embed.assert_called_once_with(doc, 128, 1, extra="param")
            assert results == [1, 2, 3]

    def test_embed_iterable(self, mock_embedding):
        """Verify embed calls iter_checked_texts and delegates to model.embed for an iterable of strings."""
        mock_embedding.model.embed.return_value = iter([10, 20])
        docs = ["doc1", "doc2"]
        mock_iter = iter(docs)

        with patch(
            "qwen3_embed.common.utils.iter_checked_texts", return_value=mock_iter
        ) as mock_iter_checked:
            results = list(mock_embedding.embed(docs, batch_size=64))

            mock_iter_checked.assert_called_once_with(docs)
            # The docs passed to model.embed should be the return value of iter_checked_texts
            mock_embedding.model.embed.assert_called_once_with(mock_iter, 64, None)
            assert results == [10, 20]

    def test_embed_default_parameters(self, mock_embedding):
        """Verify embed uses default parameters when not provided."""
        mock_embedding.model.embed.return_value = iter([])
        doc = "test"

        with patch("qwen3_embed.common.utils.check_input_length"):
            list(mock_embedding.embed(doc))

            mock_embedding.model.embed.assert_called_once_with(doc, 256, None)
