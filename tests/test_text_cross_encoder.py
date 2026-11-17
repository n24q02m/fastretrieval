import math

import numpy as np
import pytest

import fastretrieval.common.utils as utils_module
from fastretrieval.rerank.cross_encoder.qwen3_cross_encoder import Qwen3CrossEncoder
from fastretrieval.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder
from tests.onnx_runtime_stub import StubSession, install_stub_runtime, write_tokenizer_files

_YESNO_MODEL = "n24q02m/Qwen3-Reranker-0.6B-ONNX-YesNo"


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


@pytest.fixture
def stub_reranker(tmp_path, monkeypatch):
    """A real TextCrossEncoder over a YesNo graph whose "yes" logit counts "alpha" tokens."""
    model_dir = tmp_path / "model"
    vocab = write_tokenizer_files(model_dir, ["alpha", "beta", "gamma"])
    alpha_id = vocab["alpha"]

    def compute(feed: dict[str, np.ndarray]) -> np.ndarray:
        yes = (feed["input_ids"] == alpha_id).sum(axis=1).astype(np.float32)
        return np.stack([np.zeros_like(yes), yes], axis=1)

    session = StubSession(compute)
    # onnxruntime boundary: no CPython 3.15 wheel exists, so the session is a numpy stand-in.
    opened = install_stub_runtime(monkeypatch, session)
    encoder = TextCrossEncoder(
        _YESNO_MODEL, cache_dir=str(tmp_path / "cache"), specific_model_path=str(model_dir)
    )
    return encoder, session, opened


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
    from fastretrieval.rerank.cross_encoder.qwen3_cross_encoder import Qwen3CrossEncoder

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


def test_init_resolves_the_name_case_insensitively_and_forwards_backend_options(
    tmp_path, monkeypatch
):
    model_dir = tmp_path / "model"
    write_tokenizer_files(model_dir, ["alpha"])
    # onnxruntime boundary: no CPython 3.15 wheel exists, so the session is a numpy stand-in.
    opened = install_stub_runtime(monkeypatch, StubSession(lambda feed: np.zeros((1, 2))))

    encoder = TextCrossEncoder(
        _YESNO_MODEL.upper(),
        cache_dir=str(tmp_path / "cache"),
        threads=2,
        cuda=False,
        device_ids=[3],
        lazy_load=True,
        specific_model_path=str(model_dir),
    )

    backend = encoder.model
    assert type(backend) is Qwen3CrossEncoder
    assert backend.model_description.model == _YESNO_MODEL
    assert backend.threads == 2
    assert backend.cuda is False
    assert backend.device_ids == [3]
    assert backend.device_id == 3
    assert backend.lazy_load is True
    assert opened == []


def test_rerank_scores_documents_streamed_from_a_generator(stub_reranker):
    encoder, session, opened = stub_reranker

    scores = list(encoder.rerank("alpha", (doc for doc in ["alpha alpha", "beta"])))

    np.testing.assert_allclose(scores, [_sigmoid(3.0), _sigmoid(1.0)], rtol=1e-6)
    assert len(opened) == 1
    assert len(session.feeds) == 2


def test_rerank_rejects_an_oversized_document_streamed_from_a_generator(
    stub_reranker, monkeypatch
):
    encoder, _, _ = stub_reranker
    # The limit is read from the environment once at import time; lower it for this test only.
    monkeypatch.setattr(utils_module, "MAX_INPUT_LENGTH", 8)

    with pytest.raises(ValueError, match="exceeds maximum allowed length of 8 characters"):
        list(encoder.rerank("alpha", (doc for doc in ["beta", "alpha " * 3])))


def test_rerank_pairs_scores_pairs_streamed_from_a_generator(stub_reranker):
    encoder, _, _ = stub_reranker

    pairs = ((q, d) for q, d in [("alpha", "alpha"), ("beta", "gamma")])
    scores = list(encoder.rerank_pairs(pairs))

    np.testing.assert_allclose(scores, [_sigmoid(2.0), 0.5], rtol=1e-6)


@pytest.mark.parametrize(
    "pair",
    [("alpha " * 3, "beta"), ("beta", "alpha " * 3)],
    ids=["oversized-query", "oversized-document"],
)
def test_rerank_pairs_rejects_an_oversized_side_of_a_streamed_pair(
    stub_reranker, monkeypatch, pair
):
    encoder, _, _ = stub_reranker
    # The limit is read from the environment once at import time; lower it for this test only.
    monkeypatch.setattr(utils_module, "MAX_INPUT_LENGTH", 8)

    with pytest.raises(ValueError, match="exceeds maximum allowed length of 8 characters"):
        list(encoder.rerank_pairs(p for p in [("beta", "gamma"), pair]))


def test_token_count_counts_real_tokens_and_ignores_padding(stub_reranker):
    encoder, _, _ = stub_reranker

    count = encoder.token_count([("alpha beta", "gamma"), ("alpha", "beta")])

    assert count == 5
