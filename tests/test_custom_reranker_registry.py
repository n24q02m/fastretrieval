"""F5: custom reranker registry must propagate to spawned multiprocessing workers.

The embedding side already does this (CustomTextEmbedding._export_registry /
_import_registry / _extra_worker_params + the parallel pool params.update). The
reranker side did not, so TextCrossEncoder(custom).rerank_pairs(..., parallel=N)
with input >= batch_size raised "Model ... not supported" in the spawned worker
(fresh interpreter with an empty registry). These tests guard the mirror fix.
"""

import pytest

from qwen3_embed.common.model_description import BaseModelDescription, ModelSource
from qwen3_embed.rerank.cross_encoder.custom_text_cross_encoder import (
    CustomTextCrossEncoder,
    CustomTextCrossEncoderWorker,
)


def _clear() -> None:
    CustomTextCrossEncoder.SUPPORTED_MODELS.clear()


def test_custom_reranker_registry_survives_serialization():
    _clear()
    CustomTextCrossEncoder.add_model(
        BaseModelDescription(model="Org/My-Reranker", sources=ModelSource(hf="Org/My-Reranker"))
    )
    payload = CustomTextCrossEncoder._export_registry()
    _clear()
    assert CustomTextCrossEncoder.SUPPORTED_MODELS == []  # fresh-worker simulation
    CustomTextCrossEncoder._import_registry(payload)
    try:
        models = [m.model for m in CustomTextCrossEncoder._list_supported_models()]
        assert "Org/My-Reranker" in models
    finally:
        _clear()


def test_custom_reranker_import_registry_is_idempotent():
    _clear()
    desc = BaseModelDescription(model="Org/Dup", sources=ModelSource(hf="Org/Dup"))
    CustomTextCrossEncoder.add_model(desc)
    payload = CustomTextCrossEncoder._export_registry()
    CustomTextCrossEncoder._import_registry(payload)  # re-import same payload
    try:
        ids = [m.model for m in CustomTextCrossEncoder._list_supported_models()]
        assert ids.count("Org/Dup") == 1  # no duplicate entry
    finally:
        _clear()


def test_custom_reranker_extra_worker_params_carries_registry():
    _clear()
    CustomTextCrossEncoder.add_model(
        BaseModelDescription(model="Org/R", sources=ModelSource(hf="Org/R"))
    )
    try:
        enc = CustomTextCrossEncoder.__new__(CustomTextCrossEncoder)  # skip ONNX load
        params = enc._extra_worker_params()
        assert "custom_registry" in params
        assert any(d.model == "Org/R" for d in params["custom_registry"])
    finally:
        _clear()


def test_custom_reranker_uses_custom_worker_class():
    # The base OnnxTextCrossEncoder worker would construct a plain (non-custom)
    # cross-encoder + never import the registry. The custom subclass must use a
    # worker that re-registers the custom model in the spawned process.
    assert CustomTextCrossEncoder._get_worker_class() is CustomTextCrossEncoderWorker


@pytest.mark.integration
def test_custom_reranker_parallel_resolves_in_workers():
    """Register a custom reranker, rerank_pairs with parallel=2 + batch_size=1
    (forces the worker-pool path), assert no 'not supported' error + scores.

    The registered model is a plain cross-encoder, the shape CustomTextCrossEncoder
    is built for: it inherits OnnxTextCrossEncoder._post_process_onnx_output, which
    is `float(elem) for elem in model_output` and therefore requires one score per
    pair. Pointing it at a Qwen3 reranker graph instead -- as this test first did --
    hands that generic reducer the raw logits and it raises
    "only 0-dimensional arrays can be converted to Python scalars", which says
    nothing about the registry propagation this test exists to check.
    Xenova/ms-marco-MiniLM-L-6-v2 is the library's own documented example of a
    custom cross-encoder (see TextCrossEncoder.list_supported_models) and, at 80 MB,
    a seventh of the download.
    """
    _clear()
    model_id = "Org/Custom-Reranker"
    CustomTextCrossEncoder.add_model(
        BaseModelDescription(
            model=model_id,
            sources=ModelSource(hf="Xenova/ms-marco-MiniLM-L-6-v2"),
            model_file="onnx/model.onnx",
        )
    )
    try:
        from qwen3_embed.rerank.cross_encoder.text_cross_encoder import TextCrossEncoder

        ce = TextCrossEncoder(model_name=model_id)
        pairs = [("q", "d1"), ("q", "d2"), ("q", "d3")]
        scores = list(ce.rerank_pairs(pairs, batch_size=1, parallel=2))
        assert len(scores) == 3
    finally:
        _clear()
