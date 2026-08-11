import pytest

from qwen3_embed.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder


def test_init_unsupported_model_raises_value_error():
    """Initializing TextCrossEncoder with an unsupported model should raise ValueError."""
    with pytest.raises(ValueError, match="is not supported in TextCrossEncoder"):
        TextCrossEncoder(model_name="unsupported-model-name")


def test_list_supported_models():
    """Verify that list_supported_models returns a list of dictionaries with model descriptions."""
    models = TextCrossEncoder.list_supported_models()
    assert isinstance(models, list)
    assert len(models) > 0

    # Check that each item is a dictionary with expected keys
    for model in models:
        assert isinstance(model, dict)
        assert "model" in model
        assert "description" in model
        assert "size_in_GB" in model
        assert "sources" in model


def test_list_supported_models_caching(monkeypatch):
    """Verify that _list_supported_models is cached."""
    from qwen3_embed.rerank.cross_encoder.qwen3_cross_encoder import Qwen3CrossEncoder

    TextCrossEncoder._clear_model_cache()

    call_count = 0
    original_method = Qwen3CrossEncoder._list_supported_models

    def mocked_method(cls):
        nonlocal call_count
        call_count += 1
        return original_method()

    monkeypatch.setattr(Qwen3CrossEncoder, "_list_supported_models", classmethod(mocked_method))

    # First call
    TextCrossEncoder._list_supported_models()
    assert call_count == 1

    # Second call - should use cache
    TextCrossEncoder._list_supported_models()
    assert call_count == 1

    # Clear cache
    TextCrossEncoder._clear_model_cache()

    # Third call - should re-compute
    TextCrossEncoder._list_supported_models()
    assert call_count == 2
