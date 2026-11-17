import importlib.metadata

import fastretrieval


def test_public_api_is_importable_under_new_name():
    assert fastretrieval.__all__ == [
        "CustomModelSpec",
        "CustomRerankerSpec",
        "Device",
        "define_cache_dir",
        "ImageEmbedding",
        "LateInteractionTextEmbedding",
        "LateInteractionMultimodalEmbedding",
        "SparseEmbedding",
        "SparseTextEmbedding",
        "TextEmbedding",
        "TextCrossEncoder",
    ]


def test_cache_dir_is_public_api():
    from fastretrieval import define_cache_dir

    assert callable(define_cache_dir)


def test_distribution_name_is_fastretrieval():
    assert importlib.metadata.version("fastretrieval")


def test_old_module_name_is_gone():
    import pytest

    with pytest.raises(ModuleNotFoundError):
        __import__("qwen3_embed")


def test_version_falls_back_to_placeholder_without_distribution_metadata(monkeypatch, tmp_path):
    import importlib.util
    import sys
    from pathlib import Path

    # An empty search path hides the installed dist-info, as in a source checkout that was
    # never installed; submodules stay importable through the sys.modules cache.
    monkeypatch.setattr(sys, "path", [str(tmp_path)])
    spec = importlib.util.spec_from_file_location(
        "_fastretrieval_without_metadata", Path(fastretrieval.__file__)
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)

    assert module.__version__ == "0.0.0"
    assert module.version == "0.0.0"


def test_image_embedding_is_resolved_lazily_from_the_image_module():
    from fastretrieval.models.image import ImageEmbedding

    assert fastretrieval.ImageEmbedding is ImageEmbedding


def test_unknown_package_attribute_raises_attribute_error():
    import pytest

    with pytest.raises(AttributeError, match="module 'fastretrieval' has no attribute 'Missing'"):
        _ = fastretrieval.Missing
