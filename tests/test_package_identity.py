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
