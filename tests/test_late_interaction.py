import numpy as np
import pytest

from fastretrieval.common.onnx_model import OnnxOutputContext
from fastretrieval.models.late_interaction import LateInteractionTextEmbedding
from fastretrieval.models.late_interaction.colbert import Colbert


def test_colbert_post_process_masks_tokens_and_normalizes_active_rows():
    model = Colbert.__new__(Colbert)
    model.skip_list = {99}
    model.pad_token_id = 0
    output = OnnxOutputContext(
        model_output=np.array(
            [[[3.0, 4.0], [9.0, 12.0], [8.0, 15.0]]],
            dtype=np.float32,
        ),
        attention_mask=np.ones((1, 3), dtype=np.int64),
        input_ids=np.array([[10, 99, 0]], dtype=np.int64),
    )

    embeddings = list(model._post_process_onnx_output(output))

    np.testing.assert_allclose(embeddings, [np.array([[0.6, 0.8]], dtype=np.float32)])
    assert output.attention_mask.tolist() == [[1, 0, 0]]


def test_colbert_post_process_keeps_zero_norm_rows_finite_before_filtering():
    model = Colbert.__new__(Colbert)
    model.skip_list = set()
    model.pad_token_id = 0
    output = OnnxOutputContext(
        model_output=np.zeros((1, 2, 2), dtype=np.float32),
        attention_mask=np.ones((1, 2), dtype=np.int64),
        input_ids=np.array([[10, 11]], dtype=np.int64),
    )

    embeddings = list(model._post_process_onnx_output(output))

    assert len(embeddings) == 1
    assert np.isfinite(embeddings[0]).all()


def test_list_supported_models_is_not_empty():
    models = LateInteractionTextEmbedding.list_supported_models()

    assert models
    assert all("model" in model and "dim" in model for model in models)
    assert any("qwen" not in model["model"].lower() for model in models)


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="not supported"):
        LateInteractionTextEmbedding(model_name="acme/does-not-exist")
