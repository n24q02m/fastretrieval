"""Behavior of the SPLADE++ sparse stack.

Covers SpladePP, its worker, the SparseTextEmbedding facade, SparseEmbedding and
SparseTextEmbeddingBase. Every model loads from a local model directory under tmp_path that
holds a real WordPiece tokenizer and real config files; only the ONNX session is stubbed.
"""

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordPiece
from tokenizers.normalizers import BertNormalizer
from tokenizers.pre_tokenizers import BertPreTokenizer
from tokenizers.processors import TemplateProcessing

import fastretrieval.common.onnx_model as onnx_model_module
from fastretrieval.common.onnx_model import OnnxOutputContext
from fastretrieval.models.sparse import SparseEmbedding, SparseTextEmbedding
from fastretrieval.models.sparse.sparse_embedding_base import SparseTextEmbeddingBase
from fastretrieval.models.sparse.splade_pp import (
    SpladePP,
    SpladePPEmbeddingWorker,
    supported_splade_models,
)

SPLADE = "prithivida/Splade_PP_en_v1"
DEPRECATED_SPLADE = "prithvida/Splade_PP_en_v1"
VOCAB = {
    "[PAD]": 0,
    "[UNK]": 1,
    "[CLS]": 2,
    "[SEP]": 3,
    "sparse": 4,
    "retrieval": 5,
    "splade": 6,
}
PAD, CLS, SEP = VOCAB["[PAD]"], VOCAB["[CLS]"], VOCAB["[SEP]"]
SPARSE, RETRIEVAL, SPLADE_ID = VOCAB["sparse"], VOCAB["retrieval"], VOCAB["splade"]


class _StubSession:
    """Deterministic stand-in for a SPLADE masked-LM graph.

    The logit of the token sitting at position ``s`` is ``s + 1`` for its own vocabulary id and
    ``-1`` everywhere else, so the expected sparse vector is the log1p of the last position at
    which each id occurs (counting from one), restricted to attended positions.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, np.ndarray]] = []

    def get_inputs(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(name="input_ids", shape=["batch", "sequence"]),
            SimpleNamespace(name="attention_mask", shape=["batch", "sequence"]),
        ]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="logits", shape=["batch", "sequence", len(VOCAB)])]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(self, output_names: Any, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append({name: np.array(value, copy=True) for name, value in inputs.items()})
        input_ids = inputs["input_ids"]
        logits = np.full((*input_ids.shape, len(VOCAB)), -1.0, dtype=np.float32)
        rows, positions = np.indices(input_ids.shape)
        logits[rows, positions, input_ids] = positions + 1
        return [logits]


class _StubOrt:
    """Module-shaped stand-in for onnxruntime that hands out _StubSession objects."""

    GraphOptimizationLevel = SimpleNamespace(ORT_ENABLE_ALL="ORT_ENABLE_ALL")
    ExecutionMode = SimpleNamespace(ORT_PARALLEL="ORT_PARALLEL")
    SessionOptions = SimpleNamespace

    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []

    @staticmethod
    def get_available_providers() -> list[str]:
        return ["CPUExecutionProvider"]

    def InferenceSession(self, path: str, providers: Any, sess_options: Any) -> _StubSession:
        self.created.append({"path": path, "providers": providers, "sess_options": sess_options})
        return _StubSession()


@pytest.fixture
def stub_ort(monkeypatch: pytest.MonkeyPatch) -> _StubOrt:
    stub = _StubOrt()
    # Boundary (a) onnxruntime.InferenceSession: no cp315 wheel and no real graph in unit tests.
    monkeypatch.setattr(onnx_model_module, "ort", stub)
    return stub


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tokenizer = Tokenizer(WordPiece(VOCAB, unk_token="[UNK]"))
    tokenizer.normalizer = BertNormalizer(lowercase=True)
    tokenizer.pre_tokenizer = BertPreTokenizer()
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", CLS), ("[SEP]", SEP)],
    )
    tokenizer.save(str(model_dir / "tokenizer.json"))
    files = {
        "config.json": {"pad_token_id": PAD},
        "tokenizer_config.json": {"model_max_length": 32, "pad_token": "[PAD]"},
        "special_tokens_map.json": {
            "cls_token": "[CLS]",
            "pad_token": "[PAD]",
            "sep_token": "[SEP]",
            "unk_token": "[UNK]",
        },
    }
    for name, content in files.items():
        (model_dir / name).write_text(json.dumps(content), encoding="utf-8")
    return model_dir


def _splade(model_dir: Path, tmp_path: Path, **kwargs: Any) -> SpladePP:
    return SpladePP(
        SPLADE,
        cache_dir=str(tmp_path / "cache"),
        specific_model_path=str(model_dir),
        **kwargs,
    )


def _facade(
    model_dir: Path, tmp_path: Path, model_name: str = SPLADE, **kwargs: Any
) -> SparseTextEmbedding:
    return SparseTextEmbedding(
        model_name,
        cache_dir=str(tmp_path / "cache"),
        specific_model_path=str(model_dir),
        **kwargs,
    )


def _assert_sparse(embedding: SparseEmbedding, expected: dict[int, float]) -> None:
    """Compare against ``{vocabulary id: best logit}``; values are stored as log1p(logit)."""
    assert embedding.indices.tolist() == sorted(expected)
    np.testing.assert_allclose(
        embedding.values, np.log1p([expected[index] for index in sorted(expected)]), rtol=1e-6
    )


def _session(model: SpladePP) -> _StubSession:
    """Return the _StubSession that stub_ort installed as the model's ONNX session."""
    session = model.model
    assert isinstance(session, _StubSession)
    return session


# --- SpladePP: scoring ---


def test_splade_scores_each_vocabulary_id_by_log1p_of_its_best_positive_logit(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _splade(model_dir, tmp_path)

    first, second = model.embed(["sparse retrieval", "splade splade"])

    assert _session(model).calls[-1]["input_ids"].tolist() == [
        [CLS, SPARSE, RETRIEVAL, SEP],
        [CLS, SPLADE_ID, SPLADE_ID, SEP],
    ]
    _assert_sparse(first, {CLS: 1, SPARSE: 2, RETRIEVAL: 3, SEP: 4})
    _assert_sparse(second, {CLS: 1, SPLADE_ID: 3, SEP: 4})  # the repeated id keeps its max
    assert first.values.dtype == np.float32


def test_splade_ignores_logits_at_padded_positions(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _splade(model_dir, tmp_path)

    _, padded = model.embed(["sparse retrieval", "splade"])

    sent = _session(model).calls[-1]
    assert sent["input_ids"][1].tolist() == [CLS, SPLADE_ID, SEP, PAD]
    assert sent["attention_mask"][1].tolist() == [1, 1, 1, 0]
    # [PAD] has the highest logit (4) of that row, but its position is masked out.
    _assert_sparse(padded, {CLS: 1, SPLADE_ID: 2, SEP: 3})


def test_splade_post_processing_requires_an_attention_mask(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _splade(model_dir, tmp_path)
    output = OnnxOutputContext(model_output=np.zeros((1, 2, len(VOCAB)), dtype=np.float32))

    with pytest.raises(
        ValueError, match="attention_mask must be provided for document post-processing"
    ):
        list(model._post_process_onnx_output(output))


def test_splade_query_and_passage_embedding_score_text_exactly_like_embed(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _splade(model_dir, tmp_path)

    (single,) = model.query_embed("splade")
    several = list(model.query_embed(["splade", "sparse retrieval"]))
    passages = list(model.passage_embed(["splade", "sparse retrieval"]))
    documents = list(model.embed(["splade", "sparse retrieval"]))

    _assert_sparse(single, {CLS: 1, SPLADE_ID: 2, SEP: 3})
    for embeddings in (several, passages):
        assert [embedding.as_dict() for embedding in embeddings] == [
            embedding.as_dict() for embedding in documents
        ]


def test_splade_embed_loads_a_lazy_model_on_first_use(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _splade(model_dir, tmp_path, lazy_load=True)
    assert not stub_ort.created
    assert not hasattr(model, "model")

    (embedding,) = model.embed("sparse")

    assert len(stub_ort.created) == 1
    assert stub_ort.created[0]["path"] == str(model_dir / "model.onnx")
    _assert_sparse(embedding, {CLS: 1, SPARSE: 2, SEP: 3})


# --- SpladePP: configuration ---


@pytest.mark.parametrize(
    ("device_id", "device_ids", "expected"),
    [(3, [5, 6], 3), (None, [5, 6], 5), (None, None, None)],
)
def test_splade_device_id_prefers_the_explicit_id_then_the_first_worker_device(
    model_dir: Path,
    tmp_path: Path,
    device_id: int | None,
    device_ids: list[int] | None,
    expected: int | None,
) -> None:
    model = _splade(
        model_dir, tmp_path, lazy_load=True, device_id=device_id, device_ids=device_ids
    )

    assert model.device_id == expected
    assert model.device_ids == device_ids


def test_splade_forwards_exposed_session_options_and_threads_to_the_onnx_session(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _splade(model_dir, tmp_path, threads=3, enable_cpu_mem_arena=False)

    sess_options = stub_ort.created[0]["sess_options"]
    assert model._extra_session_options == {"enable_cpu_mem_arena": False}
    assert sess_options.enable_cpu_mem_arena is False
    assert sess_options.intra_op_num_threads == 3


# --- SpladePP: token counting ---


@pytest.mark.parametrize(
    ("texts", "kwargs", "expected"),
    [
        ("sparse retrieval", {}, 4),
        (["sparse", "splade retrieval"], {"batch_size": 1}, 3 + 4),
        (["sparse", "splade retrieval"], {}, 3 + 4),
    ],
)
def test_splade_token_count_counts_attended_tokens_including_cls_and_sep(
    model_dir: Path,
    tmp_path: Path,
    stub_ort: _StubOrt,
    texts: str | list[str],
    kwargs: dict[str, Any],
    expected: int,
) -> None:
    model = _splade(model_dir, tmp_path, lazy_load=True)

    assert model.token_count(texts, **kwargs) == expected
    assert len(stub_ort.created) == 1  # counting loads the lazy model for its tokenizer


# --- SpladePP: worker ---


def test_splade_worker_builds_a_single_threaded_model_that_returns_raw_logits(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    assert SpladePP._get_worker_class() is SpladePPEmbeddingWorker

    worker = SpladePPEmbeddingWorker(
        SPLADE, str(tmp_path / "cache"), specific_model_path=str(model_dir)
    )
    ((index, output),) = worker.process([(2, ["splade"])])

    assert isinstance(worker.model, SpladePP)
    assert worker.model.threads == 1
    assert stub_ort.created[0]["sess_options"].intra_op_num_threads == 1
    assert index == 2
    assert output.input_ids is not None
    assert output.input_ids.tolist() == [[CLS, SPLADE_ID, SEP]]
    assert output.model_output.shape == (1, 3, len(VOCAB))
    (embedding,) = worker.model._post_process_onnx_output(output)
    _assert_sparse(embedding, {CLS: 1, SPLADE_ID: 2, SEP: 3})


# --- SparseTextEmbedding facade ---


def test_sparse_facade_lists_both_registered_spellings_as_dicts() -> None:
    listed = SparseTextEmbedding.list_supported_models()

    assert listed == [asdict(description) for description in supported_splade_models]
    assert [entry["model"] for entry in listed] == [SPLADE, DEPRECATED_SPLADE]


def test_sparse_facade_resolves_the_canonical_name_case_insensitively_and_delegates(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    facade = _facade(model_dir, tmp_path, model_name=SPLADE.upper())

    (document,) = facade.embed(["sparse retrieval"])
    (query,) = facade.query_embed("splade")
    (passage,) = facade.passage_embed(["sparse retrieval"])

    assert isinstance(facade.model, SpladePP)
    assert facade.model.model_description.model == SPLADE
    _assert_sparse(document, {CLS: 1, SPARSE: 2, RETRIEVAL: 3, SEP: 4})
    _assert_sparse(query, {CLS: 1, SPLADE_ID: 2, SEP: 3})
    assert passage.as_dict() == document.as_dict()
    assert facade.token_count("sparse retrieval") == 4
    assert facade.token_count(["sparse", "splade"], batch_size=1) == 6


def test_sparse_facade_rejects_unregistered_models_before_loading_anything(
    tmp_path: Path, stub_ort: _StubOrt
) -> None:
    with pytest.raises(
        ValueError,
        match=r"^Model acme/none is not supported in SparseTextEmbedding\.Please check the "
        r"supported models using `SparseTextEmbedding\.list_supported_models\(\)`$",
    ):
        SparseTextEmbedding("acme/none", cache_dir=str(tmp_path / "cache"))
    assert not stub_ort.created


def test_sparse_facade_maps_the_misspelled_name_to_the_canonical_model_with_a_warning(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    with pytest.warns(
        DeprecationWarning,
        match=r"The right spelling is prithivida/Splade_PP_en_v1\. "
        r"Support of this name will be removed soon, please fix the model_name",
    ):
        facade = _facade(model_dir, tmp_path, model_name=DEPRECATED_SPLADE)

    assert facade.model_name == DEPRECATED_SPLADE
    assert isinstance(facade.model, SpladePP)
    assert facade.model.model_name == SPLADE
    assert facade.model.model_description.model == SPLADE


def test_sparse_facade_forwards_loader_options_to_splade(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    facade = _facade(
        model_dir, tmp_path, threads=2, device_ids=[4], lazy_load=True, local_files_only=True
    )

    # The base pops the flag from its own copy of kwargs, so SpladePP receives it as well.
    assert facade._local_files_only is True
    assert facade.model._local_files_only is True
    assert isinstance(facade.model, SpladePP)
    assert (facade.model.threads, facade.model.device_id, facade.model.lazy_load) == (2, 4, True)
    assert not stub_ort.created


# --- SparseEmbedding ---


def test_sparse_embedding_as_object_exposes_the_same_arrays() -> None:
    values = np.array([0.5, 0.25], dtype=np.float32)
    indices = np.array([7, 42], dtype=np.int64)

    as_object = SparseEmbedding(values=values, indices=indices).as_object()

    assert set(as_object) == {"values", "indices"}
    assert as_object["values"] is values
    assert as_object["indices"] is indices


def test_sparse_embedding_from_dict_builds_parallel_arrays_and_round_trips() -> None:
    embedding = SparseEmbedding.from_dict({42: 0.25, 7: 0.5})

    assert embedding.indices.tolist() == [42, 7]
    assert embedding.values.tolist() == [0.25, 0.5]
    assert embedding.as_dict() == {42: 0.25, 7: 0.5}


def test_sparse_embedding_from_an_empty_dict_has_no_entries() -> None:
    embedding = SparseEmbedding.from_dict({})

    assert embedding.indices.size == 0
    assert embedding.values.size == 0
    assert embedding.as_dict() == {}


# --- SparseTextEmbeddingBase ---


def test_sparse_base_stores_settings_and_leaves_model_methods_unimplemented() -> None:
    base = SparseTextEmbeddingBase(
        "acme/model", cache_dir="cache", threads=2, local_files_only=True
    )

    assert (base.model_name, base.cache_dir, base.threads) == ("acme/model", "cache", 2)
    assert base._local_files_only is True
    with pytest.raises(NotImplementedError, match="^$"):
        base.embed(["sparse"])
    with pytest.raises(NotImplementedError, match="^$"):
        list(base.passage_embed(["sparse"]))
    with pytest.raises(NotImplementedError, match="^$"):
        list(base.query_embed("sparse"))
    with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
        base.token_count("sparse")
