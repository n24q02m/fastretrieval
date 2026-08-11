"""Tests for custom model registration via TextEmbedding.add_custom_model."""

import numpy as np
import pytest

from qwen3_embed.common.model_description import (
    DenseModelDescription,
    ModelSource,
    PoolingType,
)
from qwen3_embed.common.onnx_model import OnnxOutputContext
from qwen3_embed.text.custom_text_embedding import CustomTextEmbedding
from qwen3_embed.text.text_embedding import TextEmbedding


def _ctx(rows: int, dim: int) -> OnnxOutputContext:
    out = np.ones((rows, 3, dim), dtype=np.float32)
    mask = np.ones((rows, 3), dtype=np.int64)
    return OnnxOutputContext(model_output=out, attention_mask=mask)


def test_custom_post_process_honors_dim():
    enc = CustomTextEmbedding.__new__(CustomTextEmbedding)
    enc._pooling = PoolingType.LAST_TOKEN
    enc._normalization = True
    truncated = list(enc._post_process_onnx_output(_ctx(2, 8), dim=4))
    assert all(v.shape == (4,) for v in truncated)


def test_custom_model_spec_registers_embedder():
    from qwen3_embed import CustomModelSpec

    CustomTextEmbedding._SUPPORTED.clear()
    CustomModelSpec(
        model_id="Org/gte-multilingual-base-onnx",
        hf="Org/gte-multilingual-base-onnx",
        model_file="onnx/model.onnx",
        dim=768,
        pooling="CLS",
        normalization=True,
    ).register()
    try:
        models = [m["model"].lower() for m in TextEmbedding.list_supported_models()]
        assert "org/gte-multilingual-base-onnx" in models
        desc = CustomTextEmbedding._resolve_description("Org/gte-multilingual-base-onnx")
        assert desc.pooling == PoolingType.CLS
        assert desc.dim == 768
    finally:
        CustomTextEmbedding._SUPPORTED.clear()


def test_custom_model_spec_requires_dim():
    from qwen3_embed import CustomModelSpec

    with pytest.raises(ValueError, match="dim is required"):
        CustomModelSpec(model_id="Org/no-dim", hf="Org/no-dim").register()


def test_custom_registry_survives_serialization():
    from qwen3_embed.common.model_description import (
        CustomDenseModelDescription,
        ModelSource,
        PoolingType,
    )

    CustomTextEmbedding._SUPPORTED.clear()
    desc = CustomDenseModelDescription(
        model="Org/My-Embed",
        dim=384,
        sources=ModelSource(hf="Org/My-Embed"),
        pooling=PoolingType.MEAN,
        normalization=True,
    )
    CustomTextEmbedding._register(desc)
    payload = CustomTextEmbedding._export_registry()
    CustomTextEmbedding._SUPPORTED.clear()
    CustomTextEmbedding._import_registry(payload)
    resolved = CustomTextEmbedding._resolve_description("org/my-embed")  # case-insensitive
    assert resolved.pooling == PoolingType.MEAN
    assert resolved.normalization is True
    assert resolved.dim == 384


class TestCustomModelRegistration:
    """Verify custom model registration works for all pooling types."""

    def setup_method(self, method):
        """Clear custom model registry between tests."""
        CustomTextEmbedding._SUPPORTED.clear()

    def test_register_cls_pooling_model(self):
        TextEmbedding.add_custom_model(
            model_description=DenseModelDescription(
                model="test/cls-model",
                sources=ModelSource(hf="test/cls-model"),
                dim=768,
            ),
            pooling=PoolingType.CLS,
            normalization=True,
        )
        models = TextEmbedding.list_supported_models()
        assert any(m["model"] == "test/cls-model" for m in models)

    def test_register_mean_pooling_model(self):
        TextEmbedding.add_custom_model(
            model_description=DenseModelDescription(
                model="test/mean-model",
                sources=ModelSource(hf="test/mean-model"),
                dim=512,
            ),
            pooling=PoolingType.MEAN,
            normalization=True,
        )
        models = TextEmbedding.list_supported_models()
        assert any(m["model"] == "test/mean-model" for m in models)

    def test_register_last_token_pooling_model(self):
        TextEmbedding.add_custom_model(
            model_description=DenseModelDescription(
                model="test/last-token-model",
                sources=ModelSource(hf="test/last-token-model"),
                dim=1024,
            ),
            pooling=PoolingType.LAST_TOKEN,
            normalization=True,
        )
        models = TextEmbedding.list_supported_models()
        assert any(m["model"] == "test/last-token-model" for m in models)

    def test_duplicate_model_raises(self):
        TextEmbedding.add_custom_model(
            model_description=DenseModelDescription(
                model="test/duplicate",
                sources=ModelSource(hf="test/duplicate"),
                dim=256,
            ),
            pooling=PoolingType.CLS,
            normalization=True,
        )
        with pytest.raises(ValueError, match="already registered"):
            TextEmbedding.add_custom_model(
                model_description=DenseModelDescription(
                    model="test/duplicate",
                    sources=ModelSource(hf="test/duplicate"),
                    dim=256,
                ),
                pooling=PoolingType.CLS,
                normalization=True,
            )

    def test_duplicate_model_case_insensitive_raises(self):
        """Verify that duplicate checks are case-insensitive."""
        TextEmbedding.add_custom_model(
            model_description=DenseModelDescription(
                model="test/Case-Insensitive",
                sources=ModelSource(hf="test/Case-Insensitive"),
                dim=256,
            ),
            pooling=PoolingType.CLS,
            normalization=True,
        )
        with pytest.raises(ValueError, match="already registered"):
            TextEmbedding.add_custom_model(
                model_description=DenseModelDescription(
                    model="test/case-insensitive",
                    sources=ModelSource(hf="test/case-insensitive"),
                    dim=256,
                ),
                pooling=PoolingType.CLS,
                normalization=True,
            )

    def test_conflict_with_builtin_model_raises(self):
        """Verify that registration fails if it conflicts with a built-in model."""
        builtin_model = "n24q02m/Qwen3-Embedding-0.6B-ONNX"
        with pytest.raises(ValueError, match="already registered"):
            TextEmbedding.add_custom_model(
                model_description=DenseModelDescription(
                    model=builtin_model,
                    sources=ModelSource(hf="dummy/builtin-conflict"),
                    dim=256,
                ),
                pooling=PoolingType.CLS,
                normalization=True,
            )

    def test_register_model_with_full_metadata(self):
        """Verify that all optional metadata parameters are correctly registered."""
        model_name = "test/full-metadata"
        sources = ModelSource(hf="test/full-metadata")
        additional_files = ["config.json", "vocab.txt"]
        TextEmbedding.add_custom_model(
            model_description=DenseModelDescription(
                model=model_name,
                sources=sources,
                dim=128,
                model_file="custom_model.onnx",
                description="A test model with full metadata",
                license="MIT",
                size_in_GB=1.5,
                additional_files=additional_files,
            ),
            pooling=PoolingType.MEAN,
            normalization=False,
        )

        models = TextEmbedding.list_supported_models()
        target_model = next(m for m in models if m["model"] == model_name)

        assert target_model["model"] == model_name
        assert target_model["sources"]["hf"] == "test/full-metadata"
        assert target_model["dim"] == 128
        assert target_model["model_file"] == "custom_model.onnx"
        assert target_model["description"] == "A test model with full metadata"
        assert target_model["license"] == "MIT"
        assert target_model["size_in_GB"] == 1.5
        assert target_model["additional_files"] == additional_files
