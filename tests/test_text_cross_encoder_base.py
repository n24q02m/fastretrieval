from typing import Any

import pytest

from qwen3_embed.rerank.cross_encoder.text_cross_encoder_base import TextCrossEncoderBase


class StubCrossEncoder(TextCrossEncoderBase):
    """Stub class for testing TextCrossEncoderBase."""

    @classmethod
    def _list_supported_models(cls) -> list[Any]:
        return []


def test_text_cross_encoder_base_init():
    """Verify that TextCrossEncoderBase.__init__ correctly sets attributes."""
    model_name = "test-model"
    cache_dir = "/tmp/cache"
    threads = 4
    kwargs = {"local_files_only": True, "extra_param": "value"}

    encoder = StubCrossEncoder(
        model_name=model_name, cache_dir=cache_dir, threads=threads, **kwargs
    )

    assert encoder.model_name == model_name
    assert encoder.cache_dir == cache_dir
    assert encoder.threads == threads
    assert encoder._local_files_only is True


def test_text_cross_encoder_base_not_implemented_methods():
    """Verify that abstract methods in TextCrossEncoderBase raise NotImplementedError."""
    encoder = StubCrossEncoder(model_name="test-model")

    with pytest.raises(NotImplementedError, match="should be overridden by subclasses"):
        encoder.rerank(query="query", documents=["doc1"])

    with pytest.raises(NotImplementedError, match="should be overridden by subclasses"):
        encoder.rerank_pairs(pairs=[("q", "d")])

    with pytest.raises(NotImplementedError, match="should be overridden by subclasses"):
        encoder.token_count(pairs=[("q", "d")])
