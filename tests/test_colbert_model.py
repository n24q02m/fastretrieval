"""Behavior of the ColBERT late-interaction stack.

Covers Colbert, the LateInteractionTextEmbedding facade, LateInteractionTextEmbeddingBase and
TokenEmbeddingsModel. Every model loads from a local model directory under tmp_path that holds
a real WordPiece tokenizer and real config files; only the ONNX session is stubbed.
"""

import json
import string
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
from fastretrieval.models.late_interaction import LateInteractionTextEmbedding
from fastretrieval.models.late_interaction.colbert import Colbert, ColbertEmbeddingWorker
from fastretrieval.models.late_interaction.late_interaction_embedding_base import (
    LateInteractionTextEmbeddingBase,
)
from fastretrieval.models.late_interaction.token_embeddings import (
    TokenEmbeddingsModel,
    TokensEmbeddingWorker,
)

COLBERT = "colbert-ir/colbertv2.0"
COLBERT_SMALL = "answerdotai/answerai-colbert-small-v1"
TOKENS_MODEL = "jinaai/jina-embeddings-v2-small-en-tokens"
DIM = 4
MAX_LENGTH = 48
SPECIAL_TOKENS = ["[PAD]", "[unused0]", "[unused1]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
WORDS = ["hello", "world", "late", "interaction"]
VOCAB = {
    token: index for index, token in enumerate([*SPECIAL_TOKENS, *WORDS, *string.punctuation])
}
PAD, CLS, SEP, MASK = VOCAB["[PAD]"], VOCAB["[CLS]"], VOCAB["[SEP]"], VOCAB["[MASK]"]
HELLO, WORLD, LATE, INTERACTION = (VOCAB[word] for word in WORDS)
DOC_MARKER = Colbert.DOCUMENT_MARKER_TOKEN_ID
QUERY_MARKER = Colbert.QUERY_MARKER_TOKEN_ID
QUERY_LENGTH = Colbert.MIN_QUERY_LENGTH + 1  # padded query plus the inserted query marker


class _StubSession:
    """Deterministic stand-in for a token-embedding ONNX graph.

    Output row ``[b, s]`` is ``[input_id, 1, 0, 0]``, so a test can read back which token
    produced every embedding row, before and after L2 normalization.
    """

    def __init__(self) -> None:
        self.calls: list[dict[str, np.ndarray]] = []

    def get_inputs(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(name="input_ids", shape=["batch", "sequence"]),
            SimpleNamespace(name="attention_mask", shape=["batch", "sequence"]),
        ]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="last_hidden_state", shape=["batch", "sequence", DIM])]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(self, output_names: Any, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        # Copies, because Colbert's post-processing rewrites the attention mask in place.
        self.calls.append({name: np.array(value, copy=True) for name, value in inputs.items()})
        input_ids = inputs["input_ids"]
        output = np.zeros((*input_ids.shape, DIM), dtype=np.float32)
        output[..., 0] = input_ids
        output[..., 1] = 1.0
        return [output]


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
        "tokenizer_config.json": {"model_max_length": MAX_LENGTH, "pad_token": "[PAD]"},
        "special_tokens_map.json": {
            "cls_token": "[CLS]",
            "mask_token": "[MASK]",
            "pad_token": "[PAD]",
            "sep_token": "[SEP]",
            "unk_token": "[UNK]",
        },
    }
    for name, content in files.items():
        (model_dir / name).write_text(json.dumps(content), encoding="utf-8")
    return model_dir


def _colbert(model_dir: Path, tmp_path: Path, **kwargs: Any) -> Colbert:
    return Colbert(
        COLBERT,
        cache_dir=str(tmp_path / "cache"),
        specific_model_path=str(model_dir),
        **kwargs,
    )


def _facade(model_dir: Path, tmp_path: Path, **kwargs: Any) -> LateInteractionTextEmbedding:
    return LateInteractionTextEmbedding(
        COLBERT,
        cache_dir=str(tmp_path / "cache"),
        specific_model_path=str(model_dir),
        **kwargs,
    )


def _token_ids(embedding: np.ndarray) -> list[int]:
    """Recover the token id behind each _StubSession row; the ratio survives normalization."""
    return [int(round(value)) for value in embedding[:, 0] / embedding[:, 1]]


def _session(model: Colbert) -> _StubSession:
    """Return the _StubSession that stub_ort installed as the model's ONNX session."""
    session = model.model
    assert isinstance(session, _StubSession)
    return session


# --- Colbert: loading ---


def test_colbert_load_derives_mask_pad_and_punctuation_ids_from_the_tokenizer(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _colbert(model_dir, tmp_path)

    assert model.mask_token_id == MASK
    assert model.pad_token_id == PAD
    assert model.skip_list == {VOCAB[symbol] for symbol in string.punctuation}
    assert stub_ort.created[0]["path"] == str(model_dir / "model.onnx")
    assert stub_ort.created[0]["providers"] == ["CPUExecutionProvider"]


def test_colbert_load_reserves_one_position_for_the_marker_and_pads_queries_with_mask(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _colbert(model_dir, tmp_path)

    assert model.tokenizer is not None
    assert model.query_tokenizer is not None
    assert model.tokenizer.truncation is not None
    assert model.query_tokenizer.truncation is not None
    assert model.tokenizer.truncation["max_length"] == MAX_LENGTH - 1
    assert model.query_tokenizer.truncation["max_length"] == MAX_LENGTH - 1
    padding = model.query_tokenizer.padding
    assert padding is not None
    assert padding["length"] == Colbert.MIN_QUERY_LENGTH
    assert padding["pad_id"] == MASK
    assert padding["pad_token"] == "[MASK]"


@pytest.mark.parametrize(
    ("device_id", "device_ids", "expected"),
    [(3, [5, 6], 3), (None, [5, 6], 5), (None, None, None)],
)
def test_colbert_device_id_prefers_the_explicit_id_then_the_first_worker_device(
    model_dir: Path,
    tmp_path: Path,
    device_id: int | None,
    device_ids: list[int] | None,
    expected: int | None,
) -> None:
    model = _colbert(
        model_dir, tmp_path, lazy_load=True, device_id=device_id, device_ids=device_ids
    )

    assert model.device_id == expected
    assert getattr(model, "model", None) is None


def test_colbert_forwards_exposed_session_options_to_the_onnx_session(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _colbert(model_dir, tmp_path, enable_cpu_mem_arena=False)

    assert model._extra_session_options == {"enable_cpu_mem_arena": False}
    assert stub_ort.created[0]["sess_options"].enable_cpu_mem_arena is False


# --- Colbert: documents ---


def test_colbert_embed_inserts_the_document_marker_after_cls(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _colbert(model_dir, tmp_path)

    list(model.embed(["hello, world.", "late"]))

    sent = _session(model).calls[-1]
    assert sent["input_ids"].tolist() == [
        [CLS, DOC_MARKER, HELLO, VOCAB[","], WORLD, VOCAB["."], SEP],
        [CLS, DOC_MARKER, LATE, SEP, PAD, PAD, PAD],
    ]
    assert sent["attention_mask"].tolist() == [[1] * 7, [1, 1, 1, 1, 0, 0, 0]]


def test_colbert_embed_drops_punctuation_and_padding_rows_and_normalizes_the_rest(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _colbert(model_dir, tmp_path)

    first, second = model.embed(["hello, world.", "late"])

    assert _token_ids(first) == [CLS, DOC_MARKER, HELLO, WORLD, SEP]
    assert _token_ids(second) == [CLS, DOC_MARKER, LATE, SEP]
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1.0, rtol=1e-6)
    np.testing.assert_allclose(np.linalg.norm(second, axis=1), 1.0, rtol=1e-6)


def test_colbert_truncates_long_documents_so_the_marker_still_fits_the_context(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _colbert(model_dir, tmp_path)

    (embedding,) = model.embed("hello " * (2 * MAX_LENGTH))

    assert _session(model).calls[-1]["input_ids"].shape == (1, MAX_LENGTH)
    assert embedding.shape == (MAX_LENGTH, DIM)


def test_colbert_document_post_processing_requires_input_ids_and_attention_mask(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _colbert(model_dir, tmp_path)
    output = OnnxOutputContext(model_output=np.zeros((1, 2, DIM), dtype=np.float32))

    with pytest.raises(ValueError, match="input_ids and attention_mask must be provided"):
        list(model._post_process_onnx_output(output))


# --- Colbert: queries ---


def test_colbert_query_embed_pads_short_queries_with_mask_tokens_and_keeps_raw_rows(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _colbert(model_dir, tmp_path)

    (embedding,) = model.query_embed("hello")

    assert embedding.shape == (QUERY_LENGTH, DIM)
    assert _token_ids(embedding) == [CLS, QUERY_MARKER, HELLO, SEP] + [MASK] * (QUERY_LENGTH - 4)
    assert embedding[:, 1].tolist() == [1.0] * QUERY_LENGTH  # not normalized
    sent_mask = _session(model).calls[-1]["attention_mask"][0].tolist()
    assert sent_mask == [1, 1, 1, 1] + [0] * (QUERY_LENGTH - 4)


def test_colbert_query_embed_keeps_queries_longer_than_the_minimum_unpadded(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _colbert(model_dir, tmp_path)

    (embedding,) = model.query_embed("hello " * 40)

    assert embedding.shape == (40 + 3, DIM)  # 40 words, [CLS], [SEP] and the query marker
    assert MASK not in _token_ids(embedding)


def test_colbert_query_embed_loads_a_lazy_model_and_runs_each_query_on_its_own(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _colbert(model_dir, tmp_path, lazy_load=True)
    assert not stub_ort.created

    first, second = model.query_embed(["hello", "late interaction"])

    assert len(stub_ort.created) == 1
    assert len(_session(model).calls) == 2
    assert first.shape == second.shape == (QUERY_LENGTH, DIM)
    assert _token_ids(second)[:5] == [CLS, QUERY_MARKER, LATE, INTERACTION, SEP]


# --- Colbert: token counting ---


@pytest.mark.parametrize(
    ("texts", "kwargs", "expected"),
    [
        ("hello, world.", {}, 6),
        (["hello", "late interaction"], {"include_extension": True}, 3 + 4 + 2),
        (["hello"] * 3, {"batch_size": 2, "include_extension": True}, 3 * 3 + 3),
        ("hello", {"is_doc": False}, 3),
        ("hello", {"is_doc": False, "include_extension": True}, QUERY_LENGTH),
        ("hello " * 40, {"is_doc": False, "include_extension": True}, 40 + 3),
    ],
)
def test_colbert_token_count_counts_attended_tokens_plus_optional_markers_and_mask_padding(
    model_dir: Path,
    tmp_path: Path,
    stub_ort: _StubOrt,
    texts: str | list[str],
    kwargs: dict[str, Any],
    expected: int,
) -> None:
    model = _colbert(model_dir, tmp_path, lazy_load=True)

    assert model.token_count(texts, **kwargs) == expected
    assert len(stub_ort.created) == 1  # counting loads the lazy model for its tokenizers


# --- Colbert: worker ---


def test_colbert_worker_builds_a_single_threaded_colbert_that_embeds_documents(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    assert Colbert._get_worker_class() is ColbertEmbeddingWorker

    worker = ColbertEmbeddingWorker(
        COLBERT, str(tmp_path / "cache"), specific_model_path=str(model_dir)
    )
    ((index, output),) = worker.process([(7, ["hello"])])

    assert isinstance(worker.model, Colbert)
    assert worker.model.threads == 1
    assert stub_ort.created[0]["sess_options"].intra_op_num_threads == 1
    assert index == 7
    assert output.input_ids is not None
    assert output.input_ids.tolist() == [[CLS, DOC_MARKER, HELLO, SEP]]


# --- LateInteractionTextEmbedding facade ---


def test_late_interaction_facade_resolves_colbert_case_insensitively(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    facade = LateInteractionTextEmbedding(
        COLBERT.upper(), cache_dir=str(tmp_path / "cache"), specific_model_path=str(model_dir)
    )

    assert isinstance(facade.model, Colbert)
    assert facade.model.model_description.model == COLBERT


def test_late_interaction_facade_lists_the_colbert_descriptions_as_dicts() -> None:
    listed = LateInteractionTextEmbedding.list_supported_models()

    assert listed == [asdict(description) for description in Colbert._list_supported_models()]
    assert [entry["model"] for entry in listed] == [COLBERT, COLBERT_SMALL]


def test_late_interaction_facade_rejects_unregistered_models_before_loading_anything(
    tmp_path: Path, stub_ort: _StubOrt
) -> None:
    with pytest.raises(
        ValueError,
        match=r"^Model acme/none is not supported in LateInteractionTextEmbedding\.Please check "
        r"the supported models using `LateInteractionTextEmbedding\.list_supported_models\(\)`$",
    ):
        LateInteractionTextEmbedding("acme/none", cache_dir=str(tmp_path / "cache"))
    assert not stub_ort.created


def test_late_interaction_facade_embeds_documents_passages_and_queries_through_colbert(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    facade = _facade(model_dir, tmp_path)

    (document,) = facade.embed(["hello, world."])
    (passage,) = facade.passage_embed(["hello, world."])
    (query,) = facade.query_embed("hello")

    assert _token_ids(document) == [CLS, DOC_MARKER, HELLO, WORLD, SEP]
    np.testing.assert_array_equal(passage, document)
    assert _token_ids(query)[:4] == [CLS, QUERY_MARKER, HELLO, SEP]


def test_late_interaction_facade_token_count_forwards_query_and_extension_flags(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    facade = _facade(model_dir, tmp_path)

    assert facade.token_count("hello") == 3
    assert facade.token_count("hello", is_doc=False, include_extension=True) == QUERY_LENGTH
    assert facade.token_count(["hello"] * 3, batch_size=2, include_extension=True) == 12


def test_late_interaction_facade_embedding_size_is_looked_up_once_then_cached(
    model_dir: Path, tmp_path: Path
) -> None:
    facade = LateInteractionTextEmbedding(
        COLBERT_SMALL,
        cache_dir=str(tmp_path / "cache"),
        specific_model_path=str(model_dir),
        lazy_load=True,
    )
    assert facade._embedding_size is None

    assert facade.embedding_size == 96
    assert facade._embedding_size == 96
    assert facade.embedding_size == 96


@pytest.mark.parametrize(("name", "dim"), [(COLBERT, 128), (COLBERT_SMALL.upper(), 96)])
def test_late_interaction_get_embedding_size_matches_registered_models_case_insensitively(
    name: str, dim: int
) -> None:
    assert LateInteractionTextEmbedding.get_embedding_size(name) == dim


def test_late_interaction_get_embedding_size_lists_available_models_for_unknown_names() -> None:
    with pytest.raises(
        ValueError,
        match=r"Embedding size for model acme/none was None\. Available model names: "
        r"\['colbert-ir/colbertv2\.0', 'answerdotai/answerai-colbert-small-v1'\]",
    ):
        LateInteractionTextEmbedding.get_embedding_size("acme/none")


# --- LateInteractionTextEmbeddingBase ---


def test_late_interaction_base_query_embed_embeds_one_query_or_each_query_as_documents(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    facade = _facade(model_dir, tmp_path)

    single = list(LateInteractionTextEmbeddingBase.query_embed(facade, "hello"))
    several = list(LateInteractionTextEmbeddingBase.query_embed(facade, iter(["hello", "late"])))

    assert len(single) == 1
    assert len(several) == 2
    assert _token_ids(single[0]) == [CLS, DOC_MARKER, HELLO, SEP]
    np.testing.assert_array_equal(several[0], single[0])
    assert _token_ids(several[1]) == [CLS, DOC_MARKER, LATE, SEP]


def test_late_interaction_base_stores_settings_and_leaves_model_methods_unimplemented() -> None:
    base = LateInteractionTextEmbeddingBase(
        "acme/model", cache_dir="cache", threads=2, local_files_only=True
    )

    assert (base.model_name, base.cache_dir, base.threads) == ("acme/model", "cache", 2)
    assert base._local_files_only is True
    with pytest.raises(NotImplementedError, match="^$"):
        base.embed(["hello"])
    with pytest.raises(NotImplementedError, match="^$"):
        list(base.passage_embed(["hello"]))
    with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
        LateInteractionTextEmbeddingBase.get_embedding_size("acme/model")
    with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
        _ = base.embedding_size
    with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
        base.token_count("hello")


# --- TokenEmbeddingsModel ---


def test_token_embeddings_lists_its_single_registered_model() -> None:
    (description,) = TokenEmbeddingsModel.list_supported_models()

    assert description["model"] == TOKENS_MODEL
    assert description["dim"] == 512
    assert description["model_file"] == "onnx/model.onnx"


def test_token_embeddings_yield_one_unnormalized_row_per_attended_token(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = TokenEmbeddingsModel(
        TOKENS_MODEL, cache_dir=str(tmp_path / "cache"), specific_model_path=str(model_dir)
    )

    first, second = model.embed(["hello, world", "late"])

    assert stub_ort.created[0]["path"] == str(model_dir / "onnx" / "model.onnx")
    assert first[:, 0].tolist() == [CLS, HELLO, VOCAB[","], WORLD, SEP]
    assert second[:, 0].tolist() == [CLS, LATE, SEP]
    assert first[:, 1].tolist() == [1.0] * 5


def test_token_embeddings_worker_builds_a_single_threaded_model(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    assert TokenEmbeddingsModel._get_worker_class() is TokensEmbeddingWorker

    worker = TokensEmbeddingWorker(
        TOKENS_MODEL, str(tmp_path / "cache"), specific_model_path=str(model_dir)
    )
    ((index, output),) = worker.process([(0, ["hello"])])

    assert isinstance(worker.model, TokenEmbeddingsModel)
    assert worker.model.threads == 1
    assert index == 0
    assert output.input_ids is not None
    assert output.input_ids.tolist() == [[CLS, HELLO, SEP]]
