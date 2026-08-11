"""Tests for custom model registration via TextCrossEncoder.add_custom_model."""

import pytest

from qwen3_embed.common.model_description import BaseModelDescription, ModelSource
from qwen3_embed.rerank.cross_encoder.custom_text_cross_encoder import CustomTextCrossEncoder
from qwen3_embed.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder


class TestCustomCrossEncoderRegistration:
    """Verify custom model registration for CrossEncoder works."""

    def setup_method(self, method):
        """Clear custom model registry between tests."""
        CustomTextCrossEncoder.SUPPORTED_MODELS.clear()

    def test_register_model(self):
        """Test adding a basic model."""
        TextCrossEncoder.add_custom_model(
            model_description=BaseModelDescription(
                model="test/model",
                sources=ModelSource(hf="test/model"),
            )
        )
        models = TextCrossEncoder.list_supported_models()
        assert any(m["model"] == "test/model" for m in models)

    def test_duplicate_model_raises_case_insensitive(self):
        """Test adding a duplicate model with different casing raises ValueError."""
        TextCrossEncoder.add_custom_model(
            model_description=BaseModelDescription(
                model="Test/Duplicate",
                sources=ModelSource(hf="Test/Duplicate"),
            )
        )

        # Verify it was added
        models = TextCrossEncoder.list_supported_models()
        assert any(m["model"] == "Test/Duplicate" for m in models)

        # Try adding again with different case
        with pytest.raises(ValueError, match="already registered"):
            TextCrossEncoder.add_custom_model(
                model_description=BaseModelDescription(
                    model="test/duplicate",
                    sources=ModelSource(hf="test/duplicate"),
                )
            )


class TestCustomRerankerSpec:
    """Verify the one-call CustomRerankerSpec wrapper."""

    def setup_method(self, method):
        CustomTextCrossEncoder.SUPPORTED_MODELS.clear()

    def test_spec_registers_reranker(self):
        from qwen3_embed import CustomRerankerSpec

        CustomRerankerSpec(
            model_id="Org/gte-multilingual-reranker-base",
            hf="Org/gte-multilingual-reranker-base",
            model_file="onnx/model_quantized.onnx",
        ).register()

        models = [m["model"].lower() for m in TextCrossEncoder.list_supported_models()]
        assert "org/gte-multilingual-reranker-base" in models

    def test_spec_passes_through_metadata(self):
        from qwen3_embed import CustomRerankerSpec

        CustomRerankerSpec(
            model_id="Org/full-meta-reranker",
            hf="Org/full-meta-reranker",
            model_file="custom.onnx",
            description="A test reranker",
            license="MIT",
            size_in_GB=0.34,
            additional_files=["config.json"],
        ).register()

        models = TextCrossEncoder.list_supported_models()
        target = next(m for m in models if m["model"] == "Org/full-meta-reranker")
        assert target["sources"]["hf"] == "Org/full-meta-reranker"
        assert target["model_file"] == "custom.onnx"
        assert target["description"] == "A test reranker"
        assert target["license"] == "MIT"
        assert target["size_in_GB"] == 0.34
        assert target["additional_files"] == ["config.json"]

    def test_spec_accepts_url_source(self):
        from qwen3_embed import CustomRerankerSpec

        CustomRerankerSpec(
            model_id="local/url-reranker",
            url="https://example.invalid/reranker.tar.gz",
        ).register()

        models = [m["model"] for m in TextCrossEncoder.list_supported_models()]
        assert "local/url-reranker" in models

    def test_spec_duplicate_raises(self):
        from qwen3_embed import CustomRerankerSpec

        CustomRerankerSpec(
            model_id="Org/dup-reranker",
            hf="Org/dup-reranker",
        ).register()
        with pytest.raises(ValueError, match="already registered"):
            CustomRerankerSpec(
                model_id="org/dup-reranker",
                hf="org/dup-reranker",
            ).register()
