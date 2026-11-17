"""Tests for CustomTextCrossEncoder."""

import numpy as np
import pytest

from fastretrieval.common.model_description import BaseModelDescription, ModelSource
from fastretrieval.rerank.cross_encoder.custom_text_cross_encoder import (
    CustomTextCrossEncoder,
    CustomTextCrossEncoderWorker,
)
from tests.onnx_runtime_stub import StubSession, install_stub_runtime, write_tokenizer_files


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


_WORKER_MODEL = "org/worker-model"


@pytest.fixture
def isolated_registry(monkeypatch):
    """Give the custom registry a private, empty model list and a fresh description cache."""
    monkeypatch.setattr(CustomTextCrossEncoder, "SUPPORTED_MODELS", [])
    CustomTextCrossEncoder._clear_model_cache()
    yield CustomTextCrossEncoder.SUPPORTED_MODELS
    CustomTextCrossEncoder._clear_model_cache()


@pytest.fixture
def stub_model_dir(tmp_path, monkeypatch):
    """A real tokenizer directory plus a stub graph scoring each pair by its "alpha" count."""
    model_dir = tmp_path / "model"
    vocab = write_tokenizer_files(model_dir, ["alpha", "beta", "gamma"])
    alpha_id = vocab["alpha"]

    def compute(feed: dict[str, np.ndarray]) -> np.ndarray:
        return (feed["input_ids"] == alpha_id).sum(axis=1, keepdims=True).astype(np.float32)

    # onnxruntime boundary: no CPython 3.15 wheel exists, so the session is a numpy stand-in.
    install_stub_runtime(monkeypatch, StubSession(compute))
    return model_dir


def _worker_description() -> BaseModelDescription:
    return BaseModelDescription(
        model=_WORKER_MODEL,
        sources=ModelSource(hf=_WORKER_MODEL),
        model_file="onnx/model.onnx",
        description="",
        license="",
        size_in_GB=0.0,
    )


def test_worker_reregisters_the_shipped_registry_and_scores_pairs(
    isolated_registry, stub_model_dir, tmp_path
):
    payload = [_worker_description()]

    worker = CustomTextCrossEncoderWorker(
        _WORKER_MODEL,
        str(tmp_path / "cache"),
        custom_registry=payload,
        specific_model_path=str(stub_model_dir),
    )
    ((idx, output),) = list(worker.process([(4, [("alpha", "alpha beta"), ("beta", "gamma")])]))

    assert CustomTextCrossEncoder._get_worker_class() is CustomTextCrossEncoderWorker
    assert [desc.model for desc in isolated_registry] == [_WORKER_MODEL]
    assert type(worker.model) is CustomTextCrossEncoder
    assert worker.model.threads == 1
    assert idx == 4
    np.testing.assert_array_equal(output.model_output, [2.0, 0.0])


def test_worker_without_a_registry_payload_uses_the_models_already_registered(
    isolated_registry, stub_model_dir, tmp_path
):
    CustomTextCrossEncoder.add_model(_worker_description())

    worker = CustomTextCrossEncoderWorker(
        _WORKER_MODEL, str(tmp_path / "cache"), specific_model_path=str(stub_model_dir)
    )

    assert isinstance(worker.model, CustomTextCrossEncoder)
    assert worker.model.model_description.model == _WORKER_MODEL
    assert len(isolated_registry) == 1
    assert worker.model._extra_worker_params() == {"custom_registry": [_worker_description()]}


def test_import_registry_skips_models_already_present_in_any_case(isolated_registry):
    original = _worker_description()
    shouted = BaseModelDescription(
        model=_WORKER_MODEL.upper(),
        sources=ModelSource(hf=_WORKER_MODEL),
        model_file="onnx/model.onnx",
        description="",
        license="",
        size_in_GB=0.0,
    )

    CustomTextCrossEncoder._import_registry([original, shouted])
    CustomTextCrossEncoder._import_registry([original])

    assert isolated_registry == [original]
