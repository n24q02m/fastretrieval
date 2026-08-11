"""Tests for CustomTextEmbedding — covers custom_text_embedding.py."""

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from qwen3_embed.common.model_description import (
    CustomDenseModelDescription,
    DenseModelDescription,
    ModelSource,
    PoolingType,
)
from qwen3_embed.common.onnx_model import OnnxOutputContext
from qwen3_embed.common.types import NumpyArray
from qwen3_embed.text.custom_text_embedding import CustomTextEmbedding

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_NAME = "test-org/custom-embed-model"
_MODEL_DESC = DenseModelDescription(
    model=_MODEL_NAME,
    sources=ModelSource(hf=_MODEL_NAME),
    model_file="onnx/model.onnx",
    description="test custom embedding",
    license="MIT",
    size_in_GB=0.1,
    dim=4,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_embeddings(shape: tuple[int, ...], dtype=np.float32) -> NumpyArray:
    return np.ones(shape, dtype=dtype)


def _make_attention_mask(batch: int, seq_len: int, pad_from: int = -1) -> np.ndarray:
    """Create attention mask with 1s up to pad_from and 0s after.

    If pad_from is -1 (default), all tokens are marked as real (full mask).
    """
    if pad_from == -1:
        pad_from = seq_len
    mask = np.zeros((batch, seq_len), dtype=np.int64)
    mask[:, :pad_from] = 1
    return mask


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_registry():
    """Isolate the class-level registry between tests."""
    original = dict(CustomTextEmbedding._SUPPORTED)
    yield
    CustomTextEmbedding._SUPPORTED.clear()
    CustomTextEmbedding._SUPPORTED.update(original)


def _register(
    model_name: str = _MODEL_NAME,
    pooling: PoolingType = PoolingType.CLS,
    normalization: bool = True,
) -> CustomDenseModelDescription:
    desc = CustomDenseModelDescription(
        model=model_name,
        sources=ModelSource(hf=model_name),
        model_file="onnx/model.onnx",
        description="test",
        license="MIT",
        size_in_GB=0.1,
        dim=4,
        pooling=pooling,
        normalization=normalization,
    )
    CustomTextEmbedding._register(desc)
    return desc


def _build(
    tmp_path: Path,
    model_name: str = _MODEL_NAME,
    pooling: PoolingType = PoolingType.CLS,
    normalization: bool = True,
    **kwargs: Any,
) -> CustomTextEmbedding:
    """Build a CustomTextEmbedding with download + load mocked out."""
    _register(model_name=model_name, pooling=pooling, normalization=normalization)
    with (
        patch.object(CustomTextEmbedding, "download_model", return_value=tmp_path),
        patch.object(CustomTextEmbedding, "load_onnx_model"),
    ):
        return CustomTextEmbedding(model_name=model_name, lazy_load=False, **kwargs)


# ===========================================================================
# CustomDenseModelDescription dataclass
# ===========================================================================


class TestCustomDenseModelDescription:
    def test_frozen_dataclass(self) -> None:
        desc = CustomDenseModelDescription(
            model="org/model",
            sources=ModelSource(hf="org/model"),
            dim=4,
            pooling=PoolingType.CLS,
            normalization=True,
        )
        assert desc.pooling == PoolingType.CLS
        assert desc.normalization is True

    def test_immutable(self) -> None:
        desc = CustomDenseModelDescription(
            model="org/model",
            sources=ModelSource(hf="org/model"),
            dim=4,
            pooling=PoolingType.MEAN,
            normalization=False,
        )
        with pytest.raises((AttributeError, TypeError)):
            desc.pooling = PoolingType.CLS  # type: ignore[misc]


# ===========================================================================
# CustomTextEmbedding._register — classmethod
# ===========================================================================


class TestRegister:
    def test_adds_to_supported_registry(self) -> None:
        desc = _register(pooling=PoolingType.CLS)
        assert desc in CustomTextEmbedding._SUPPORTED.values()

    def test_carries_pooling_and_normalization(self) -> None:
        _register(pooling=PoolingType.MEAN, normalization=False)
        resolved = CustomTextEmbedding._resolve_description(_MODEL_NAME)
        assert resolved.pooling == PoolingType.MEAN
        assert resolved.normalization is False

    def test_multiple_models_independent(self) -> None:
        _register(model_name="org/model-a", pooling=PoolingType.CLS, normalization=True)
        _register(model_name="org/model-b", pooling=PoolingType.LAST_TOKEN, normalization=False)
        assert CustomTextEmbedding._resolve_description("org/model-a").pooling == PoolingType.CLS
        assert (
            CustomTextEmbedding._resolve_description("org/model-b").pooling
            == PoolingType.LAST_TOKEN
        )


# ===========================================================================
# CustomTextEmbedding._list_supported_models (lines 58-59)
# ===========================================================================


class TestListSupportedModels:
    def test_returns_supported_models(self) -> None:
        desc = _register()
        models = CustomTextEmbedding._list_supported_models()
        assert desc in models

    def test_empty_when_none_registered(self) -> None:
        # _reset_registry restores original list; after clearing there's none
        models = CustomTextEmbedding._list_supported_models()
        # At module load, SUPPORTED_MODELS is [], so only registered ones appear
        assert isinstance(models, list)


# ===========================================================================
# CustomTextEmbedding.__init__ (lines 42-55)
# ===========================================================================


class TestCustomTextEmbeddingInit:
    def test_sets_pooling_and_normalization_cls(self, tmp_path: Path) -> None:
        emb = _build(tmp_path, pooling=PoolingType.CLS, normalization=True)
        assert emb._pooling == PoolingType.CLS
        assert emb._normalization is True

    def test_sets_pooling_mean_no_norm(self, tmp_path: Path) -> None:
        emb = _build(tmp_path, pooling=PoolingType.MEAN, normalization=False)
        assert emb._pooling == PoolingType.MEAN
        assert emb._normalization is False

    def test_sets_pooling_last_token(self, tmp_path: Path) -> None:
        emb = _build(tmp_path, pooling=PoolingType.LAST_TOKEN, normalization=True)
        assert emb._pooling == PoolingType.LAST_TOKEN

    def test_sets_pooling_disabled(self, tmp_path: Path) -> None:
        emb = _build(tmp_path, pooling=PoolingType.DISABLED, normalization=False)
        assert emb._pooling == PoolingType.DISABLED

    def test_inherits_model_name(self, tmp_path: Path) -> None:
        emb = _build(tmp_path, pooling=PoolingType.CLS)
        assert emb.model_name == _MODEL_NAME

    def test_lazy_load_does_not_call_load(self, tmp_path: Path) -> None:
        _register(pooling=PoolingType.CLS)
        called: list[bool] = []
        with (
            patch.object(CustomTextEmbedding, "download_model", return_value=tmp_path),
            patch.object(
                CustomTextEmbedding, "load_onnx_model", side_effect=lambda: called.append(True)
            ),
        ):
            CustomTextEmbedding(model_name=_MODEL_NAME, lazy_load=True)
        assert not called

    def test_custom_worker_returns_correct_type(self, tmp_path: Path) -> None:
        from qwen3_embed.text.custom_text_embedding import CustomTextEmbeddingWorker

        _register(pooling=PoolingType.CLS)
        with (
            patch.object(CustomTextEmbedding, "download_model", return_value=tmp_path),
            patch.object(CustomTextEmbedding, "load_onnx_model"),
        ):
            worker = CustomTextEmbeddingWorker(_MODEL_NAME, str(tmp_path))
            emb = worker.init_embedding(_MODEL_NAME, str(tmp_path))
        assert isinstance(emb, CustomTextEmbedding)


# ===========================================================================
# CustomTextEmbedding._pool (lines 66-89)
# ===========================================================================


class TestPool:
    def setup_method(self) -> None:
        """Use a minimal registered instance; avoid real ONNX."""
        # We test _pool directly — we can instantiate with mocks.
        pass

    def _make_instance(self, pooling: PoolingType, tmp_path: Path) -> CustomTextEmbedding:
        return _build(tmp_path, pooling=pooling, normalization=True)

    def test_cls_returns_first_token(self, tmp_path: Path) -> None:
        emb = self._make_instance(PoolingType.CLS, tmp_path)
        # shape (2, 5, 4) => CLS = [:, 0] = shape (2, 4)
        embeddings = np.arange(40, dtype=np.float32).reshape(2, 5, 4)
        result = emb._pool(embeddings, attention_mask=None)
        np.testing.assert_array_equal(result, embeddings[:, 0])
        assert result.shape == (2, 4)

    def test_mean_pooling_with_mask(self, tmp_path: Path) -> None:
        emb = self._make_instance(PoolingType.MEAN, tmp_path)
        embeddings = np.ones((2, 4, 8), dtype=np.float32)
        mask = _make_attention_mask(batch=2, seq_len=4, pad_from=3)
        result = emb._pool(embeddings, attention_mask=mask)
        assert result.shape == (2, 8)

    def test_mean_pooling_no_mask_raises(self, tmp_path: Path) -> None:
        emb = self._make_instance(PoolingType.MEAN, tmp_path)
        with pytest.raises(ValueError, match="attention_mask must be provided"):
            emb._pool(np.ones((2, 4, 8), dtype=np.float32), attention_mask=None)

    def test_last_token_pooling_with_mask(self, tmp_path: Path) -> None:
        emb = self._make_instance(PoolingType.LAST_TOKEN, tmp_path)
        # batch=1, seq=4, dim=8; real tokens at positions 0,1,2 (last=2)
        embeddings = np.arange(32, dtype=np.float32).reshape(1, 4, 8)
        # right-padding: [1,1,1,0] — last real token is index 2
        mask = np.array([[1, 1, 1, 0]], dtype=np.int64)
        result = emb._pool(embeddings, attention_mask=mask)
        assert result.shape == (1, 8)

    def test_last_token_pooling_no_mask_raises(self, tmp_path: Path) -> None:
        emb = self._make_instance(PoolingType.LAST_TOKEN, tmp_path)
        with pytest.raises(ValueError, match="attention_mask must be provided"):
            emb._pool(np.ones((1, 4, 8), dtype=np.float32), attention_mask=None)

    def test_disabled_returns_unchanged(self, tmp_path: Path) -> None:
        emb = self._make_instance(PoolingType.DISABLED, tmp_path)
        embeddings = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        result = emb._pool(embeddings, attention_mask=None)
        np.testing.assert_array_equal(result, embeddings)

    def test_unsupported_pooling_raises(self, tmp_path: Path) -> None:
        emb = self._make_instance(PoolingType.CLS, tmp_path)
        # Force an unsupported pooling type at runtime
        object.__setattr__(emb, "_pooling", "UNSUPPORTED")  # bypass normal attr setting
        with pytest.raises(ValueError, match="Unsupported pooling type"):
            emb._pool(np.ones((1, 4, 8), dtype=np.float32), attention_mask=None)


# ===========================================================================
# CustomTextEmbedding._normalize (line 92)
# ===========================================================================


class TestNormalize:
    def test_normalize_true(self, tmp_path: Path) -> None:
        emb = _build(tmp_path, pooling=PoolingType.CLS, normalization=True)
        vecs = np.array([[3.0, 4.0]], dtype=np.float32)  # norm = 5
        result = emb._normalize(vecs)
        np.testing.assert_allclose(np.linalg.norm(result, axis=1), [1.0], atol=1e-6)

    def test_normalize_false_returns_unchanged(self, tmp_path: Path) -> None:
        emb = _build(tmp_path, pooling=PoolingType.CLS, normalization=False)
        vecs = np.array([[3.0, 4.0]], dtype=np.float32)
        result = emb._normalize(vecs)
        np.testing.assert_array_equal(result, vecs)


# ===========================================================================
# CustomTextEmbedding._post_process_onnx_output (line 64)
# ===========================================================================


class TestPostProcessOnnxOutput:
    def _output(self, batch: int, seq_len: int, dim: int) -> OnnxOutputContext:
        embeddings = np.ones((batch, seq_len, dim), dtype=np.float32)
        mask = _make_attention_mask(batch=batch, seq_len=seq_len, pad_from=seq_len)
        return OnnxOutputContext(model_output=embeddings, attention_mask=mask)

    def test_cls_pooling_normalized(self, tmp_path: Path) -> None:
        emb = _build(tmp_path, pooling=PoolingType.CLS, normalization=True)
        ctx = self._output(2, 5, 4)
        result = np.asarray(emb._post_process_onnx_output(ctx))
        # shape should be (2, 4) — CLS token from each row, normalized
        assert result.shape == (2, 4)
        # Each row should be normalized to unit length
        norms = np.linalg.norm(result, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), atol=1e-6)

    def test_mean_pooling_with_mask(self, tmp_path: Path) -> None:
        emb = _build(tmp_path, pooling=PoolingType.MEAN, normalization=False)
        ctx = self._output(2, 4, 8)
        result = np.asarray(emb._post_process_onnx_output(ctx))
        assert result.shape == (2, 8)

    def test_last_token_pooling(self, tmp_path: Path) -> None:
        emb = _build(tmp_path, pooling=PoolingType.LAST_TOKEN, normalization=False)
        ctx = self._output(1, 4, 6)
        result = np.asarray(emb._post_process_onnx_output(ctx))
        assert result.shape == (1, 6)

    def test_disabled_pooling(self, tmp_path: Path) -> None:
        emb = _build(tmp_path, pooling=PoolingType.DISABLED, normalization=False)
        embeddings = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        mask = _make_attention_mask(2, 3)
        ctx = OnnxOutputContext(model_output=embeddings, attention_mask=mask)
        result = np.asarray(emb._post_process_onnx_output(ctx))
        np.testing.assert_array_equal(result, embeddings)

    def test_normalization_applied_after_pool(self, tmp_path: Path) -> None:
        emb = _build(tmp_path, pooling=PoolingType.CLS, normalization=True)
        # Non-unit-norm embeddings at CLS position
        embeddings = np.array([[[3.0, 4.0, 0.0, 0.0]]], dtype=np.float32)  # (1, 1, 4)
        mask = np.ones((1, 1), dtype=np.int64)
        ctx = OnnxOutputContext(model_output=embeddings, attention_mask=mask)
        result = np.asarray(emb._post_process_onnx_output(ctx))
        # After norm, L2 should be 1.0
        np.testing.assert_allclose(np.linalg.norm(result), 1.0, atol=1e-6)

    def test_no_normalization(self, tmp_path: Path) -> None:
        emb = _build(tmp_path, pooling=PoolingType.CLS, normalization=False)
        embeddings = np.array([[[3.0, 4.0, 0.0, 0.0]]], dtype=np.float32)  # (1, 1, 4)
        mask = np.ones((1, 1), dtype=np.int64)
        ctx = OnnxOutputContext(model_output=embeddings, attention_mask=mask)
        result = np.asarray(emb._post_process_onnx_output(ctx))
        # Unnormalized: L2 should be 5.0
        np.testing.assert_allclose(np.linalg.norm(result), 5.0, atol=1e-6)
