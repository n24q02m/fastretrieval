import pytest

from qwen3_embed.common.model_description import DenseModelDescription, ModelSource


def test_model_source_valid_hf():
    """Test ModelSource with only hf parameter set."""
    source = ModelSource(hf="org/model")
    assert source.hf == "org/model"
    assert source.url is None


def test_model_source_valid_url():
    """Test ModelSource with only url parameter set."""
    source = ModelSource(url="https://example.com/model.tar.gz")
    assert source.url == "https://example.com/model.tar.gz"
    assert source.hf is None


def test_model_source_valid_both():
    """Test ModelSource with both hf and url parameters set."""
    source = ModelSource(hf="org/model", url="https://example.com/model.tar.gz")
    assert source.hf == "org/model"
    assert source.url == "https://example.com/model.tar.gz"


def test_model_source_invalid_none():
    """Test ModelSource raises ValueError when neither hf nor url is set."""
    with pytest.raises(ValueError, match="At least one source should be set"):
        ModelSource()


def test_dense_model_description_valid_dim():
    """Test DenseModelDescription with valid dim."""
    desc = DenseModelDescription(
        model="test-model",
        sources=ModelSource(hf="org/model"),
        model_file="model.onnx",
        description="Test description",
        license="MIT",
        size_in_GB=1.0,
        dim=768,
    )
    assert desc.dim == 768


def test_dense_model_description_invalid_dim():
    """Test DenseModelDescription raises ValueError when dim is None."""
    with pytest.raises(ValueError, match="dim is required for dense model description"):
        DenseModelDescription(
            model="test-model",
            sources=ModelSource(hf="org/model"),
            model_file="model.onnx",
            description="Test description",
            license="MIT",
            size_in_GB=1.0,
            dim=None,
        )


def test_model_source_deprecated_tar_struct():
    """Test ModelSource deprecated_tar_struct property."""
    # Test default value
    source_default = ModelSource(hf="org/model")
    assert source_default.deprecated_tar_struct is False

    # Test explicitly set value
    source_deprecated = ModelSource(hf="org/model", _deprecated_tar_struct=True)
    assert source_deprecated.deprecated_tar_struct is True
