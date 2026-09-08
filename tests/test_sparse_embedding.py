import numpy as np
import pytest

from fastretrieval.common.onnx_model import OnnxOutputContext
from fastretrieval.models.sparse import SparseEmbedding, SparseTextEmbedding
from fastretrieval.models.sparse.splade_pp import SpladePP


def test_splade_post_process_matches_masked_log1p_max_contract():
    model = SpladePP.__new__(SpladePP)
    output = OnnxOutputContext(
        model_output=np.array(
            [
                [[1.0, -2.0, 0.5], [4.0, 3.0, -1.0], [100.0, 100.0, 100.0]],
                [[-1.0, -2.0, -3.0], [9.0, 2.0, -4.0], [8.0, 8.0, 8.0]],
            ],
            dtype=np.float32,
        ),
        attention_mask=np.array([[1, 1, 0], [0, 0, 0]], dtype=np.int64),
    )

    embeddings = list(model._post_process_onnx_output(output))

    assert len(embeddings) == 2
    np.testing.assert_array_equal(embeddings[0].indices, [0, 1, 2])
    np.testing.assert_allclose(
        embeddings[0].values,
        np.log1p([4.0, 3.0, 0.5]).astype(np.float32),
    )
    assert embeddings[1].indices.size == 0
    assert embeddings[1].values.size == 0


def test_sparse_embedding_as_dict_roundtrip():
    emb = SparseEmbedding(values=np.array([0.5, 0.25]), indices=np.array([7, 42]))
    assert emb.as_dict() == {7: 0.5, 42: 0.25}


def test_list_supported_models_is_not_empty():
    models = SparseTextEmbedding.list_supported_models()
    assert models
    assert all("model" in model and "vocab_size" in model for model in models)
    assert any("qwen" not in model["model"].lower() for model in models)


def test_unknown_model_raises_with_available_names():
    with pytest.raises(ValueError, match="not supported"):
        SparseTextEmbedding(model_name="acme/does-not-exist")
