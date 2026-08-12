import pytest

from fastretrieval.models.late_interaction import LateInteractionTextEmbedding


def test_list_supported_models_is_not_empty():
    models = LateInteractionTextEmbedding.list_supported_models()

    assert models
    assert all("model" in model and "dim" in model for model in models)
    assert any("qwen" not in model["model"].lower() for model in models)


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="not supported"):
        LateInteractionTextEmbedding(model_name="acme/does-not-exist")
