"""Unit tests for Qwen3TextEmbedding model registration, configuration, and post-processing."""

import inspect

import numpy as np
import pytest

from qwen3_embed.common.onnx_model import OnnxOutputContext
from qwen3_embed.text.qwen3_embedding import (
    DEFAULT_TASK,
    QUERY_INSTRUCTION_TEMPLATE,
    Qwen3TextEmbedding,
    supported_qwen3_models,
)
from qwen3_embed.text.text_embedding import TextEmbedding


class TestQwen3TextEmbeddingRegistry:
    """Verify Qwen3 embedding models are properly registered."""

    def test_qwen3_in_registry(self):
        """Qwen3TextEmbedding should be in the TextEmbedding registry."""
        assert Qwen3TextEmbedding in TextEmbedding.EMBEDDINGS_REGISTRY

    def test_qwen3_models_listed(self):
        """Qwen3 models should appear in list_supported_models."""
        models = TextEmbedding.list_supported_models()
        qwen3_models = [m for m in models if "Qwen3" in m["model"]]
        assert len(qwen3_models) >= 1

    def test_qwen3_model_description(self):
        """Verify Qwen3-Embedding-0.6B model description fields."""
        desc = supported_qwen3_models[0]
        assert desc.model == "n24q02m/Qwen3-Embedding-0.6B-ONNX"
        assert desc.dim == 1024
        assert desc.license == "apache-2.0"
        assert "last-token" in desc.description.lower() or "MRL" in desc.description

    def test_default_model_is_qwen3(self):
        """TextEmbedding default model should be Qwen3."""
        sig = inspect.signature(TextEmbedding.__init__)
        default_model = sig.parameters["model_name"].default
        assert default_model == "n24q02m/Qwen3-Embedding-0.6B-ONNX"


class TestQwen3InstructionFormat:
    """Verify instruction template formatting."""

    def test_query_instruction_format(self):
        result = QUERY_INSTRUCTION_TEMPLATE.format(task="Find docs", text="test query")
        assert result == "Instruct: Find docs\nQuery: test query"

    def test_default_task_not_empty(self):
        assert len(DEFAULT_TASK) > 0
        assert "retrieve" in DEFAULT_TASK.lower()


class TestQwen3PostProcessing:
    """Test _post_process_onnx_output with mocked ONNX output."""

    def test_post_process_last_token_pooling(self):
        """Should extract last-token embeddings and normalise."""
        model_output = np.array(
            [
                [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.0, 0.0, 0.0, 0.0]],
                [[0.9, 1.0, 1.1, 1.2], [1.3, 1.4, 1.5, 1.6], [1.7, 1.8, 1.9, 2.0]],
            ]
        )
        attention_mask = np.array([[1, 1, 0], [1, 1, 1]], dtype=np.int64)
        output = OnnxOutputContext(model_output=model_output, attention_mask=attention_mask)

        result = np.array(
            list(Qwen3TextEmbedding._post_process_onnx_output(None, output))  # type: ignore
        )

        assert result.shape == (2, 4)
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, [1.0, 1.0], atol=1e-6)

    def test_post_process_mrl_truncation(self):
        """Passing dim= should truncate before normalisation."""
        model_output = np.array([[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]])
        attention_mask = np.array([[1, 1]], dtype=np.int64)
        output = OnnxOutputContext(model_output=model_output, attention_mask=attention_mask)

        result = np.array(
            list(Qwen3TextEmbedding._post_process_onnx_output(None, output, dim=2))  # type: ignore
        )
        assert result.shape == (1, 2)
        np.testing.assert_allclose(np.linalg.norm(result[0]), 1.0, atol=1e-6)

    def test_post_process_raises_without_attention_mask(self):
        """Should raise if attention_mask is None."""
        output = OnnxOutputContext(model_output=np.zeros((1, 3, 4)), attention_mask=None)
        with pytest.raises(ValueError, match="attention_mask"):
            list(Qwen3TextEmbedding._post_process_onnx_output(None, output))  # type: ignore
