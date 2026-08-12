import numpy as np
import pytest

from fastretrieval.models.sparse import SparseEmbedding, SparseTextEmbedding


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
