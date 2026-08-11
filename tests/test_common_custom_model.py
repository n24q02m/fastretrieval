from unittest.mock import patch

import pytest

from qwen3_embed.common.custom_model import CustomModelSpec, CustomRerankerSpec
from qwen3_embed.common.model_description import PoolingType


class TestCustomModelSpec:
    def test_initialization(self):
        spec = CustomModelSpec(
            model_id="test-model", hf="test-hf", dim=768, pooling="MEAN", normalization=True
        )
        assert spec.model_id == "test-model"
        assert spec.hf == "test-hf"
        assert spec.dim == 768
        assert spec.pooling == "MEAN"
        assert spec.normalization is True

    def test_register_success(self):
        spec = CustomModelSpec(
            model_id="test-model",
            hf="test-hf",
            dim=768,
            pooling=PoolingType.CLS,
            normalization=False,
            additional_files=["file1.txt"],
        )
        with patch("qwen3_embed.TextEmbedding.add_custom_model") as mock_add:
            spec.register()
            mock_add.assert_called_once()
            args, kwargs = mock_add.call_args
            description = args[0]
            assert description.model == "test-model"
            assert description.dim == 768
            assert description.sources.hf == "test-hf"
            assert description.additional_files == ["file1.txt"]
            assert kwargs["pooling"] == PoolingType.CLS
            assert kwargs["normalization"] is False

    def test_register_missing_dim_raises_error(self):
        spec = CustomModelSpec(model_id="test-model", hf="test-hf")
        with pytest.raises(ValueError, match="dim is required"):
            spec.register()

    def test_register_with_url(self):
        spec = CustomModelSpec(
            model_id="test-model", url="https://example.com/model.onnx", dim=384
        )
        with patch("qwen3_embed.TextEmbedding.add_custom_model") as mock_add:
            spec.register()
            args, _ = mock_add.call_args
            assert args[0].sources.url == "https://example.com/model.onnx"


class TestCustomRerankerSpec:
    def test_initialization(self):
        spec = CustomRerankerSpec(
            model_id="test-reranker",
            hf="test-hf-reranker",
            description="test description",
            license="MIT",
            size_in_GB=0.5,
        )
        assert spec.model_id == "test-reranker"
        assert spec.hf == "test-hf-reranker"
        assert spec.description == "test description"
        assert spec.license == "MIT"
        assert spec.size_in_GB == 0.5

    def test_register_success(self):
        spec = CustomRerankerSpec(
            model_id="test-reranker",
            hf="test-hf-reranker",
            model_file="custom.onnx",
            additional_files=["extra.json"],
        )
        with patch("qwen3_embed.TextCrossEncoder.add_custom_model") as mock_add:
            spec.register()
            mock_add.assert_called_once()
            args, _ = mock_add.call_args
            description = args[0]
            assert description.model == "test-reranker"
            assert description.sources.hf == "test-hf-reranker"
            assert description.model_file == "custom.onnx"
            assert description.additional_files == ["extra.json"]

    def test_register_with_url(self):
        spec = CustomRerankerSpec(
            model_id="test-reranker", url="https://example.com/reranker.onnx"
        )
        with patch("qwen3_embed.TextCrossEncoder.add_custom_model") as mock_add:
            spec.register()
            args, _ = mock_add.call_args
            assert args[0].sources.url == "https://example.com/reranker.onnx"
