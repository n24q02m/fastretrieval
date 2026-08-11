"""Tests for CustomTextCrossEncoder."""

from qwen3_embed.common.model_description import BaseModelDescription, ModelSource
from qwen3_embed.rerank.cross_encoder.custom_text_cross_encoder import CustomTextCrossEncoder


class TestCustomTextCrossEncoder:
    """Tests for CustomTextCrossEncoder custom model registry methods."""

    def setup_method(self) -> None:
        """Clear custom model registry between tests."""
        CustomTextCrossEncoder.SUPPORTED_MODELS.clear()

    def test_list_supported_models_empty_initially(self) -> None:
        """Verify the model list is empty initially."""
        models = CustomTextCrossEncoder._list_supported_models()
        assert isinstance(models, list)
        assert len(models) == 0

    def test_add_model(self) -> None:
        """Verify add_model correctly adds a model description."""
        desc = BaseModelDescription(
            model="org/test-model",
            sources=ModelSource(hf="org/test-model"),
            model_file="onnx/model.onnx",
            description="",
            license="",
            size_in_GB=0.0,
        )
        CustomTextCrossEncoder.add_model(desc)

        models = CustomTextCrossEncoder._list_supported_models()
        assert len(models) == 1
        assert models[0].model == "org/test-model"

    def test_add_multiple_models(self) -> None:
        """Verify multiple models can be added and listed."""
        desc1 = BaseModelDescription(
            model="org/test-model-1",
            sources=ModelSource(hf="org/test-model-1"),
            model_file="onnx/model.onnx",
            description="",
            license="",
            size_in_GB=0.0,
        )
        desc2 = BaseModelDescription(
            model="org/test-model-2",
            sources=ModelSource(hf="org/test-model-2"),
            model_file="onnx/model.onnx",
            description="",
            license="",
            size_in_GB=0.0,
        )

        CustomTextCrossEncoder.add_model(desc1)
        CustomTextCrossEncoder.add_model(desc2)

        models = CustomTextCrossEncoder._list_supported_models()
        assert len(models) == 2
        assert models[0].model == "org/test-model-1"
        assert models[1].model == "org/test-model-2"

    def test_custom_text_cross_encoder_init_passes_args(self, tmp_path) -> None:
        """Verify that CustomTextCrossEncoder passes arguments correctly during initialization."""
        # Create a dummy model directory to avoid actual downloading
        model_dir = tmp_path / "dummy_model"
        model_dir.mkdir()

        # Add a test model to the registry
        desc = BaseModelDescription(
            model="org/init-test-model",
            sources=ModelSource(hf="org/init-test-model"),
            model_file="onnx/model.onnx",
            description="",
            license="",
            size_in_GB=0.0,
        )
        CustomTextCrossEncoder.add_model(desc)

        # We patch load_onnx_model so we don't actually try to load a real ONNX file
        # and download_model so we don't try to download from HF
        from unittest.mock import patch

        with (
            patch.object(CustomTextCrossEncoder, "download_model", return_value=model_dir),
            patch.object(CustomTextCrossEncoder, "load_onnx_model") as mock_load,
        ):
            encoder = CustomTextCrossEncoder(
                model_name="org/init-test-model",
                threads=4,
            )

            assert encoder.model_name == "org/init-test-model"
            assert encoder.threads == 4
            mock_load.assert_called_once()
