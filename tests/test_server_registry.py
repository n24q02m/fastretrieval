"""Tests for the model-name registry and session cache."""

from __future__ import annotations

import pytest

from fastretrieval.server import registry


@pytest.fixture(autouse=True)
def _fresh_sessions():
    registry.reset_sessions()
    yield
    registry.reset_sessions()


def test_resolve_embed_alias() -> None:
    assert registry.resolve_model("qwen3-0.6b", "embed") == "n24q02m/Qwen3-Embedding-0.6B-ONNX"


def test_resolve_rerank_alias() -> None:
    assert (
        registry.resolve_model("qwen3-rerank-0.6b", "rerank") == "n24q02m/Qwen3-Reranker-0.6B-ONNX"
    )


def test_resolve_is_case_insensitive() -> None:
    assert registry.resolve_model("QWEN3-0.6B", "embed") == "n24q02m/Qwen3-Embedding-0.6B-ONNX"


def test_resolve_passthrough_for_canonical_names() -> None:
    canonical = "n24q02m/Qwen3-Embedding-0.6B-ONNX-Q4F16"
    assert registry.resolve_model(canonical, "embed") == canonical
    # The same string in the other endpoint's table also passes through.
    assert registry.resolve_model("custom/model", "rerank") == "custom/model"


def test_every_alias_target_is_a_registered_model() -> None:
    """An alias that names an unregistered model turns every request into a 400."""
    from fastretrieval.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder
    from fastretrieval.text.text_embedding import TextEmbedding

    embed_names = {m["model"] for m in TextEmbedding.list_supported_models()}
    rerank_names = {m["model"] for m in TextCrossEncoder.list_supported_models()}
    assert set(registry.EMBED_MODELS.values()) <= embed_names
    assert set(registry.RERANK_MODELS.values()) <= rerank_names


def test_gguf_alias_resolves_to_gguf_artifact() -> None:
    session = registry.get_backend("qwen3-0.6b-gguf", "embed")
    assert session.model_name == "n24q02m/Qwen3-Embedding-0.6B-GGUF"


def test_onnx_alias_resolves_to_onnx_artifact() -> None:
    session = registry.get_backend("qwen3-0.6b", "embed")
    assert session.model_name == "n24q02m/Qwen3-Embedding-0.6B-ONNX"


def test_sessions_are_cached_per_resolved_name() -> None:
    a = registry.get_backend("qwen3-0.6b", "embed")
    b = registry.get_backend("QWEN3-0.6B", "embed")
    c = registry.get_backend("qwen3-0.6b-gguf", "embed")
    assert a is b
    assert a is not c


def test_sessions_are_scoped_by_kind() -> None:
    """The same model name must not share a session across embed and rerank."""
    embed = registry.get_backend("n24q02m/Qwen3-Embedding-0.6B-ONNX", "embed")
    rerank = registry.get_backend("n24q02m/Qwen3-Embedding-0.6B-ONNX", "rerank")
    assert embed is not rerank
    assert (embed.kind, rerank.kind) == ("embed", "rerank")


def test_explicit_cache_dir_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    session = registry.Session(
        "n24q02m/Qwen3-Embedding-0.6B-ONNX", "embed", cache_dir="/tmp/fastretrieval-models"
    )
    assert session.cache_dir == "/tmp/fastretrieval-models"


def test_cache_dir_left_to_the_library(monkeypatch: pytest.MonkeyPatch) -> None:
    """get_backend leaves cache_dir unset so define_cache_dir resolves FASTRETRIEVAL_CACHE_PATH."""
    monkeypatch.setenv("FASTRETRIEVAL_CACHE_PATH", "/tmp/from-env")
    session = registry.get_backend("qwen3-0.6b", "embed")
    assert session.cache_dir is None


def test_threads_env_is_forwarded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTRETRIEVAL_THREADS", "4")
    session = registry.get_backend("qwen3-0.6b", "embed")
    assert session.threads == 4


def test_invalid_threads_env_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTRETRIEVAL_THREADS", "banana")
    session = registry.get_backend("qwen3-0.6b", "embed")
    assert session.threads is None


def test_non_positive_threads_env_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FASTRETRIEVAL_THREADS", "0")
    session = registry.get_backend("qwen3-0.6b", "embed")
    assert session.threads is None


def test_check_dims_bounds() -> None:
    assert registry.check_dims(registry.DIM_MIN) == registry.DIM_MIN
    assert registry.check_dims(registry.DIM_MAX) == registry.DIM_MAX
    for bad in (registry.DIM_MIN - 1, registry.DIM_MAX + 1):
        with pytest.raises(ValueError, match="dims"):
            registry.check_dims(bad)


def test_ranked_indices_is_stable_on_ties() -> None:
    assert registry.ranked_indices([0.5, 0.5, 0.9]) == [(2, 0.9), (0, 0.5), (1, 0.5)]


def test_ranked_indices_top_n() -> None:
    assert registry.ranked_indices([0.2, 0.9, 0.5], 2) == [(1, 0.9), (2, 0.5)]


FR_ALIASES = {
    "minilm-l6-v2": "n24q02m/fr-all-MiniLM-L6-v2",
    "minilm-l12-v2": "n24q02m/fr-all-MiniLM-L12-v2",
    "multi-qa-minilm-l6-cos-v1": "n24q02m/fr-multi-qa-MiniLM-L6-cos-v1",
    "paraphrase-minilm-l3-v2": "n24q02m/fr-paraphrase-MiniLM-L3-v2",
    "paraphrase-multilingual-minilm-l12-v2": "n24q02m/fr-paraphrase-multilingual-MiniLM-L12-v2",
}


def test_fr_aliases_resolve_to_hosted_repos() -> None:
    for alias, repo in FR_ALIASES.items():
        assert registry.resolve_model(alias, "embed") == repo


def test_fr_repos_are_registered_models() -> None:
    from fastretrieval.text.text_embedding import TextEmbedding

    supported = {model["model"] for model in TextEmbedding.list_supported_models()}
    assert set(FR_ALIASES.values()) <= supported
