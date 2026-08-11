"""Tests for GGUF Text Embedding (Qwen3TextEmbeddingGGUF)."""

from __future__ import annotations

import itertools
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qwen3_embed.text.gguf_embedding import (
    DEFAULT_TASK,
    QUERY_INSTRUCTION_TEMPLATE,
    Qwen3TextEmbeddingGGUF,
    _check_llama_cpp,
)

# Note: qwen3_embed.text.gguf_embedding imports llama_cpp lazily inside
# functions (see _check_llama_cpp / Qwen3TextEmbeddingGGUF.__init__), not at
# module import time, so no sys.modules mock is needed just to import it.
# Each test below that needs llama_cpp to appear "installed" scopes its own
# mock via patch.dict(sys.modules, {"llama_cpp": ...}) so nothing leaks into
# other test modules collected in the same pytest session (in particular,
# tests/test_integration_gguf.py relies on llama_cpp being genuinely absent
# so its pytest.importorskip("llama_cpp") can skip when the optional
# dependency is not installed).

# ---------------------------------------------------------------------------
# Tests for _check_llama_cpp
# ---------------------------------------------------------------------------


def test_check_llama_cpp_missing():
    """Test that _check_llama_cpp raises ImportError when llama_cpp is absent."""
    with (
        patch.dict(sys.modules, {"llama_cpp": None}),
        pytest.raises(ImportError) as excinfo,
    ):
        _check_llama_cpp()

    # Verify message and exception chaining (raise ... from e)
    assert "llama-cpp-python is required for GGUF models" in str(excinfo.value)
    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_check_llama_cpp_import_error_with_mock():
    """Test that _check_llama_cpp raises ImportError from builtins.__import__."""
    original_import = __import__

    def mock_import(name, *args, **kwargs):
        if name == "llama_cpp":
            raise ImportError("Mocked import error for llama_cpp")
        return original_import(name, *args, **kwargs)

    with (
        patch("builtins.__import__", side_effect=mock_import),
        pytest.raises(ImportError) as excinfo,
    ):
        _check_llama_cpp()

    assert "llama-cpp-python is required for GGUF models" in str(excinfo.value)
    assert excinfo.value.__cause__ is not None
    assert "Mocked import error for llama_cpp" in str(excinfo.value.__cause__)


def test_check_llama_cpp_exact_message():
    """Test that _check_llama_cpp raises ImportError with the exact expected message."""
    expected_msg = (
        "llama-cpp-python is required for GGUF models. Install with: pip install qwen3-embed[gguf]"
    )
    with (
        patch.dict(sys.modules, {"llama_cpp": None}),
        pytest.raises(ImportError) as excinfo,
    ):
        _check_llama_cpp()

    assert str(excinfo.value) == expected_msg


def test_check_llama_cpp_present():
    """Test that _check_llama_cpp does not raise when llama_cpp is present."""
    mock_module = MagicMock()
    with patch.dict(sys.modules, {"llama_cpp": mock_module}):
        _check_llama_cpp()  # Should not raise


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_llm(embedding_dim: int = 8) -> MagicMock:
    """Return a mock Llama instance whose create_embedding returns vectors for each input."""
    mock_llm = MagicMock()
    vec = np.random.rand(embedding_dim).astype(np.float32).tolist()

    def mock_create_embedding(docs, *args, **kwargs):
        if isinstance(docs, str):
            docs = [docs]
        return {"data": [{"embedding": vec} for _ in docs]}

    mock_llm.create_embedding.side_effect = mock_create_embedding
    return mock_llm


def _make_model(embedding_dim: int = 8, **extra_attrs: Any) -> Qwen3TextEmbeddingGGUF:
    """Create a GGUF model instance with __init__ fully bypassed."""
    with patch(
        "qwen3_embed.text.gguf_embedding.Qwen3TextEmbeddingGGUF.__init__", return_value=None
    ):
        model = Qwen3TextEmbeddingGGUF()
    model._llm = _make_mock_llm(embedding_dim)
    for k, v in extra_attrs.items():
        setattr(model, k, v)
    return model


# ---------------------------------------------------------------------------
# Tests for __init__
# ---------------------------------------------------------------------------


class TestGGUFEmbeddingInit:
    def test_init_creates_llm_with_cpu(self, tmp_path: Path):
        """Test __init__ creates Llama with n_gpu_layers=0 for CPU device."""
        from qwen3_embed.common.types import Device

        # Create the expected GGUF file so model_path.exists() returns True
        model_file = tmp_path / "qwen3-embedding-0.6b-q4-k-m.gguf"
        model_file.touch()

        mock_llama_cls = MagicMock()
        mock_llama_module = MagicMock()
        mock_llama_module.Llama = mock_llama_cls

        with (
            patch.dict(sys.modules, {"llama_cpp": mock_llama_module}),
            patch.object(Qwen3TextEmbeddingGGUF, "download_model", return_value=str(tmp_path)),
            patch("qwen3_embed.text.gguf_embedding.define_cache_dir", return_value=Path("/tmp")),
        ):
            model = Qwen3TextEmbeddingGGUF(
                model_name="n24q02m/Qwen3-Embedding-0.6B-GGUF",
                cuda=Device.CPU,
            )

        mock_llama_cls.assert_called_once()
        call_kwargs = mock_llama_cls.call_args[1]
        assert call_kwargs["n_gpu_layers"] == 0
        assert call_kwargs["embedding"] is True
        assert call_kwargs["verbose"] is False
        assert model._llm is mock_llama_cls.return_value

    def test_init_creates_llm_with_gpu_auto(self, tmp_path: Path):
        """Test __init__ creates Llama with n_gpu_layers=-1 for Device.AUTO."""
        from qwen3_embed.common.types import Device

        model_file = tmp_path / "qwen3-embedding-0.6b-q4-k-m.gguf"
        model_file.touch()

        mock_llama_cls = MagicMock()
        mock_llama_module = MagicMock()
        mock_llama_module.Llama = mock_llama_cls

        with (
            patch.dict(sys.modules, {"llama_cpp": mock_llama_module}),
            patch.object(Qwen3TextEmbeddingGGUF, "download_model", return_value=str(tmp_path)),
            patch("qwen3_embed.text.gguf_embedding.define_cache_dir", return_value=Path("/tmp")),
        ):
            Qwen3TextEmbeddingGGUF(
                model_name="n24q02m/Qwen3-Embedding-0.6B-GGUF",
                cuda=Device.AUTO,
            )

        call_kwargs = mock_llama_cls.call_args[1]
        assert call_kwargs["n_gpu_layers"] == -1

    def test_init_creates_llm_with_cuda_false(self, tmp_path: Path):
        """Test __init__ creates Llama with n_gpu_layers=0 when cuda=False."""
        model_file = tmp_path / "qwen3-embedding-0.6b-q4-k-m.gguf"
        model_file.touch()

        mock_llama_cls = MagicMock()
        mock_llama_module = MagicMock()
        mock_llama_module.Llama = mock_llama_cls

        with (
            patch.dict(sys.modules, {"llama_cpp": mock_llama_module}),
            patch.object(Qwen3TextEmbeddingGGUF, "download_model", return_value=str(tmp_path)),
            patch("qwen3_embed.text.gguf_embedding.define_cache_dir", return_value=Path("/tmp")),
        ):
            Qwen3TextEmbeddingGGUF(
                model_name="n24q02m/Qwen3-Embedding-0.6B-GGUF",
                cuda=False,
            )

        call_kwargs = mock_llama_cls.call_args[1]
        assert call_kwargs["n_gpu_layers"] == 0

    def test_init_raises_if_gguf_file_missing(self, tmp_path: Path):
        """Test __init__ raises FileNotFoundError if GGUF file does not exist."""
        mock_llama_module = MagicMock()

        with (
            patch.dict(sys.modules, {"llama_cpp": mock_llama_module}),
            patch.object(Qwen3TextEmbeddingGGUF, "download_model", return_value=str(tmp_path)),
            patch("qwen3_embed.text.gguf_embedding.define_cache_dir", return_value=Path("/tmp")),
            pytest.raises(FileNotFoundError, match="GGUF model file not found"),
        ):
            Qwen3TextEmbeddingGGUF(model_name="n24q02m/Qwen3-Embedding-0.6B-GGUF")

    def test_init_raises_if_llama_cpp_missing(self):
        """Test __init__ raises ImportError if llama_cpp is not installed."""
        with (
            patch.dict(sys.modules, {"llama_cpp": None}),
            pytest.raises(ImportError, match="llama-cpp-python is required"),
        ):
            Qwen3TextEmbeddingGGUF(model_name="n24q02m/Qwen3-Embedding-0.6B-GGUF")

    def test_init_threads_default_zero(self, tmp_path: Path):
        """Test __init__ uses threads=0 (auto-detect) when threads=None."""
        model_file = tmp_path / "qwen3-embedding-0.6b-q4-k-m.gguf"
        model_file.touch()

        mock_llama_cls = MagicMock()
        mock_llama_module = MagicMock()
        mock_llama_module.Llama = mock_llama_cls

        with (
            patch.dict(sys.modules, {"llama_cpp": mock_llama_module}),
            patch.object(Qwen3TextEmbeddingGGUF, "download_model", return_value=str(tmp_path)),
            patch("qwen3_embed.text.gguf_embedding.define_cache_dir", return_value=Path("/tmp")),
        ):
            Qwen3TextEmbeddingGGUF(model_name="n24q02m/Qwen3-Embedding-0.6B-GGUF", threads=None)

        call_kwargs = mock_llama_cls.call_args[1]
        assert call_kwargs["n_threads"] == 0

    def test_init_threads_custom(self, tmp_path: Path):
        """Test __init__ passes custom threads count to Llama."""
        model_file = tmp_path / "qwen3-embedding-0.6b-q4-k-m.gguf"
        model_file.touch()

        mock_llama_cls = MagicMock()
        mock_llama_module = MagicMock()
        mock_llama_module.Llama = mock_llama_cls

        with (
            patch.dict(sys.modules, {"llama_cpp": mock_llama_module}),
            patch.object(Qwen3TextEmbeddingGGUF, "download_model", return_value=str(tmp_path)),
            patch("qwen3_embed.text.gguf_embedding.define_cache_dir", return_value=Path("/tmp")),
        ):
            Qwen3TextEmbeddingGGUF(model_name="n24q02m/Qwen3-Embedding-0.6B-GGUF", threads=4)

        call_kwargs = mock_llama_cls.call_args[1]
        assert call_kwargs["n_threads"] == 4


# ---------------------------------------------------------------------------
# Tests for embed
# ---------------------------------------------------------------------------


class TestGGUFEmbeddingEmbed:
    def test_embed_single_string(self):
        """Test embed with a single string returns one normalized embedding."""
        model = _make_model(embedding_dim=4)
        results = list(model.embed("hello"))

        assert len(results) == 1
        vec = results[0]
        assert isinstance(vec, np.ndarray)
        assert vec.dtype == np.float32
        # Check L2 normalized
        np.testing.assert_allclose(np.linalg.norm(vec), 1.0, atol=1e-5)

    def test_embed_list_of_strings(self):
        """Test embed with a list of strings returns one embedding per document."""
        model = _make_model(embedding_dim=4)
        docs = ["doc1", "doc2", "doc3"]
        results = list(model.embed(docs))

        assert len(results) == 3
        assert model._llm.create_embedding.call_count == 1

    def test_embed_mrl_truncation(self):
        """Test embed with dim kwarg truncates to requested dimension."""
        model = _make_model(embedding_dim=8)
        results = list(model.embed("hello", dim=4))

        assert len(results) == 1
        assert results[0].shape == (4,)

    def test_embed_no_mrl_returns_full_dim(self):
        """Test embed without dim returns full embedding dimension."""
        model = _make_model(embedding_dim=8)
        results = list(model.embed("hello"))

        assert results[0].shape == (8,)

    def test_embed_l2_normalization(self):
        """Test embed returns L2 normalized embeddings."""
        model = _make_model(embedding_dim=4)

        # Override create_embedding to return a known vector
        def custom_create_embedding(docs, *args, **kwargs):
            if isinstance(docs, str):
                docs = [docs]
            return {"data": [{"embedding": [3.0, 4.0, 0.0, 0.0]} for _ in docs]}

        model._llm.create_embedding.side_effect = custom_create_embedding
        results = list(model.embed("test"))
        # norm([3, 4, 0, 0]) = 5, normalized = [0.6, 0.8, 0, 0]
        np.testing.assert_allclose(results[0], [0.6, 0.8, 0.0, 0.0], atol=1e-5)

    def test_embed_zero_vector_not_divided(self):
        """Test embed with zero vector does not divide by zero."""
        model = _make_model(embedding_dim=4)

        def zero_create_embedding(docs, *args, **kwargs):
            if isinstance(docs, str):
                docs = [docs]
            return {"data": [{"embedding": [0.0, 0.0, 0.0, 0.0]} for _ in docs]}

        model._llm.create_embedding.side_effect = zero_create_embedding
        results = list(model.embed("zero"))
        # Zero vector stays zero
        np.testing.assert_array_equal(results[0], [0.0, 0.0, 0.0, 0.0])

    def test_embed_calls_create_embedding_with_document(self):
        """Test embed passes the document string to create_embedding."""
        model = _make_model(embedding_dim=4)
        list(model.embed("specific text"))
        model._llm.create_embedding.assert_called_once_with(["specific text"])

    def test_embed_large_batch(self):
        """Test embed with a large number of generated strings to ensure batching logic is correct."""
        model = _make_model(embedding_dim=4)

        # Override create_embedding to return a constant vector
        def custom_create_embedding(docs, *args, **kwargs):
            if isinstance(docs, str):
                docs = [docs]
            return {"data": [{"embedding": [3.0, 4.0, 0.0, 0.0]} for _ in docs]}

        model._llm.create_embedding.side_effect = custom_create_embedding

        num_docs = 1000
        docs = [f"document {i}" for i in range(num_docs)]

        results = list(model.embed(docs, batch_size=32, parallel=4))

        assert len(results) == num_docs
        assert model._llm.create_embedding.call_count == 32

        # Verify calls were made with correct documents
        calls = [c[0][0] for c in model._llm.create_embedding.call_args_list]
        import itertools

        flattened_calls = list(itertools.chain.from_iterable(calls))
        assert flattened_calls == docs

    def test_embed_generator_input(self):
        """Test embed handles generator iterables correctly."""
        model = _make_model(embedding_dim=4)

        def custom_create_embedding(docs, *args, **kwargs):
            if isinstance(docs, str):
                docs = [docs]
            return {"data": [{"embedding": [3.0, 4.0, 0.0, 0.0]} for _ in docs]}

        model._llm.create_embedding.side_effect = custom_create_embedding

        def doc_generator():
            for i in range(5):
                yield f"gen doc {i}"

        results = list(model.embed(doc_generator()))

        assert len(results) == 5
        assert model._llm.create_embedding.call_count == 1

        calls = [c[0][0] for c in model._llm.create_embedding.call_args_list]
        expected = [f"gen doc {i}" for i in range(5)]
        flattened_calls = list(itertools.chain.from_iterable(calls))
        assert flattened_calls == expected


# ---------------------------------------------------------------------------
# Tests for query_embed
# ---------------------------------------------------------------------------


class TestGGUFEmbeddingQueryEmbed:
    @pytest.fixture
    def model(self):
        """Create a GGUF model instance with mocked dependencies."""
        with patch(
            "qwen3_embed.text.gguf_embedding.Qwen3TextEmbeddingGGUF.__init__", return_value=None
        ):
            model = Qwen3TextEmbeddingGGUF()
            # Mock the embed method since query_embed calls it
            model.embed = MagicMock()  # type: ignore[invalid-assignment]
            return model

    def test_query_embed_single_string_default_task(self, model):
        """Test query_embed with a single string and default task."""
        query = "test query"
        list(model.query_embed(query))

        expected_text = QUERY_INSTRUCTION_TEMPLATE.format(task=DEFAULT_TASK, text=query)
        model.embed.assert_called_once()
        call_args = model.embed.call_args
        # embed receives a list of strings even for a single query
        assert list(call_args[0][0]) == [expected_text]

    def test_query_embed_list_default_task(self, model):
        """Test query_embed with a list of strings and default task."""
        queries = ["query1", "query2"]
        list(model.query_embed(queries))

        expected_texts = [
            QUERY_INSTRUCTION_TEMPLATE.format(task=DEFAULT_TASK, text=q) for q in queries
        ]
        model.embed.assert_called_once()
        call_args = model.embed.call_args
        assert list(call_args[0][0]) == expected_texts

    def test_query_embed_custom_task(self, model):
        """Test query_embed with a custom task."""
        query = "test query"
        custom_task = "Custom retrieval task"
        list(model.query_embed(query, task=custom_task))

        expected_text = QUERY_INSTRUCTION_TEMPLATE.format(task=custom_task, text=query)
        model.embed.assert_called_once()
        call_args = model.embed.call_args
        assert list(call_args[0][0]) == [expected_text]
        # Ensure task is popped from kwargs and not passed to embed
        assert "task" not in call_args[1]

    def test_query_embed_passes_kwargs(self, model):
        """Test query_embed passes other kwargs (like dim) to embed."""
        query = "test query"
        list(model.query_embed(query, dim=128, other_arg="value"))

        model.embed.assert_called_once()
        call_args = model.embed.call_args
        assert call_args[1]["dim"] == 128
        assert call_args[1]["other_arg"] == "value"


# ---------------------------------------------------------------------------
# Tests for passage_embed
# ---------------------------------------------------------------------------


class TestGGUFEmbeddingPassageEmbed:
    def test_passage_embed_delegates_to_embed(self):
        """Test passage_embed delegates to embed without instruction prefix."""
        model = _make_model(embedding_dim=4)
        texts = ["passage 1", "passage 2"]
        results = list(model.passage_embed(texts))

        assert len(results) == 2
        # Verify create_embedding was called with raw text (no instruction prefix)
        calls = [c[0][0] for c in model._llm.create_embedding.call_args_list]
        flattened_calls = list(itertools.chain.from_iterable(calls))
        assert flattened_calls == ["passage 1", "passage 2"]

    def test_passage_embed_with_dim(self):
        """Test passage_embed supports dim kwarg for MRL truncation."""
        model = _make_model(embedding_dim=8)
        results = list(model.passage_embed(["text"], dim=4))

        assert results[0].shape == (4,)

    def test_passage_embed_normalized(self):
        """Test passage_embed returns L2 normalized results."""
        model = _make_model(embedding_dim=4)
        results = list(model.passage_embed(["some text"]))

        np.testing.assert_allclose(np.linalg.norm(results[0]), 1.0, atol=1e-5)
