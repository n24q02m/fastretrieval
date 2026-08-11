"""Tests for OnnxTextModel and OnnxTextEmbedding — covers onnx_text_model.py and onnx_embedding.py."""

from collections.abc import Iterable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qwen3_embed.common.model_description import DenseModelDescription, ModelSource
from qwen3_embed.common.onnx_model import OnnxOutputContext, OnnxSessionConfig
from qwen3_embed.common.types import NumpyArray
from qwen3_embed.text.onnx_embedding import (
    OnnxTextEmbedding,
    OnnxTextEmbeddingWorker,
    supported_onnx_models,
)
from qwen3_embed.text.onnx_text_model import OnnxTextModel, TextEmbeddingWorker

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MODEL_NAME = "test-org/test-onnx-model"
_MODEL_DESC = DenseModelDescription(
    model=_MODEL_NAME,
    sources=ModelSource(hf="test-org/test-onnx-model"),
    model_file="onnx/model.onnx",
    description="test",
    license="MIT",
    size_in_GB=0.1,
    dim=4,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_tokenizer(n_docs: int = 1, seq_len: int = 4) -> MagicMock:
    ids = [1, 2, 3, 0][:seq_len]
    mask = [1, 1, 1, 0][:seq_len]
    enc = MagicMock()
    enc.ids = ids
    enc.attention_mask = mask
    tok = MagicMock()
    tok.encode_batch.return_value = [enc] * n_docs
    return tok


def _make_mock_session(output_shape: tuple[int, ...] = (1, 4), dtype=np.float32) -> MagicMock:
    output = np.ones(output_shape, dtype=dtype)
    session = MagicMock()
    session.run.return_value = [output]
    node_ids = MagicMock()
    node_ids.name = "input_ids"
    node_attn = MagicMock()
    node_attn.name = "attention_mask"
    session.get_inputs.return_value = [node_ids, node_attn]
    return session


@pytest.fixture()
def registered_test_model():
    """Add the test model to OnnxTextEmbedding.supported_onnx_models."""
    supported_onnx_models.append(_MODEL_DESC)
    yield _MODEL_DESC
    if _MODEL_DESC in supported_onnx_models:
        supported_onnx_models.remove(_MODEL_DESC)


# ---------------------------------------------------------------------------
# Concrete OnnxTextModel subclass for unit-testing the base class
# ---------------------------------------------------------------------------


class ConcreteOnnxTextModel(OnnxTextModel[NumpyArray]):
    """Minimal concrete subclass — does not load any real models."""

    @classmethod
    def _get_worker_class(cls) -> type["TextEmbeddingWorker[NumpyArray]"]:
        return _StubWorker

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[NumpyArray]:
        yield output.model_output

    def load_onnx_model(self) -> None:
        pass  # no-op in tests


class _StubWorker(TextEmbeddingWorker[NumpyArray]):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> ConcreteOnnxTextModel:
        m = ConcreteOnnxTextModel()
        m.model = _make_mock_session()
        m.model_input_names = {"input_ids", "attention_mask"}
        m.tokenizer = _make_mock_tokenizer()
        return m


# ===========================================================================
# OnnxTextModel — base class tests  (covers onnx_text_model.py)
# ===========================================================================


class TestOnnxTextModelNotImplemented:
    """Lines 23, 35, 72 — abstract methods raise NotImplementedError."""

    def test_get_worker_class_raises(self) -> None:
        class _Stub(OnnxTextModel):
            pass

        with pytest.raises(NotImplementedError):
            _Stub._get_worker_class()

    def test_post_process_raises(self) -> None:
        class _Stub(OnnxTextModel):
            pass

        stub = _Stub()
        with pytest.raises(NotImplementedError):
            stub._post_process_onnx_output(OnnxOutputContext(model_output=np.zeros((1, 4))))

    def test_load_onnx_model_raises(self) -> None:
        class _Stub(OnnxTextModel):
            pass

        stub = _Stub()
        with pytest.raises(NotImplementedError):
            stub.load_onnx_model()


class TestOnnxTextModelInit:
    """Lines 38-40 — __init__ initialises tokenizer and special_token_to_id."""

    def test_initial_state(self) -> None:
        m = ConcreteOnnxTextModel()
        assert m.tokenizer is None
        assert m.special_token_to_id == {}
        assert m.model is None


class TestOnnxTextModelPreprocessInput:
    """Line 48 — _preprocess_onnx_input returns input unchanged."""

    def test_returns_identity(self) -> None:
        m = ConcreteOnnxTextModel()
        data: dict[str, NumpyArray] = {"input_ids": np.array([[1, 2, 3]])}
        assert m._preprocess_onnx_input(data) is data


class TestOnnxTextModelTokenize:
    """Lines 75-76 — tokenize calls encode_batch."""

    def test_delegates_to_tokenizer(self) -> None:
        m = ConcreteOnnxTextModel()
        m.tokenizer = _make_mock_tokenizer(n_docs=2)
        result = m.tokenize(["hello", "world"])
        m.tokenizer.encode_batch.assert_called_once_with(["hello", "world"])
        assert len(result) == 2

    def test_asserts_tokenizer_not_none(self) -> None:
        m = ConcreteOnnxTextModel()
        with pytest.raises(AssertionError):
            m.tokenize(["hello"])


class TestOnnxTextModelOnnxEmbed:
    """Lines 83-107 — onnx_embed constructs ONNX inputs and returns OnnxOutputContext."""

    def setup_method(self) -> None:
        self.m = ConcreteOnnxTextModel()
        self.m.model = _make_mock_session((1, 4))
        self.m.model_input_names = {"input_ids", "attention_mask"}
        self.m.tokenizer = _make_mock_tokenizer(n_docs=1)

    def test_basic_context_fields(self) -> None:
        ctx = self.m.onnx_embed(["hello"])
        assert ctx.model_output is not None
        assert ctx.attention_mask is not None
        assert ctx.input_ids is not None

    def test_no_model_raises(self) -> None:
        m = ConcreteOnnxTextModel()
        with pytest.raises(ValueError, match="Model not loaded"):
            m.onnx_embed(["hello"])

    def test_includes_token_type_ids(self) -> None:
        m = ConcreteOnnxTextModel()
        session = MagicMock()
        session.run.return_value = [np.ones((1, 4), dtype=np.float32)]
        node_ids = MagicMock()
        node_ids.name = "input_ids"
        node_attn = MagicMock()
        node_attn.name = "attention_mask"
        node_tti = MagicMock()
        node_tti.name = "token_type_ids"
        session.get_inputs.return_value = [node_ids, node_attn, node_tti]
        m.model = session
        m.model_input_names = {"input_ids", "attention_mask", "token_type_ids"}
        m.tokenizer = _make_mock_tokenizer()

        m.onnx_embed(["hello"])
        call_args = session.run.call_args[0]
        onnx_input_dict = call_args[1]
        assert "token_type_ids" in onnx_input_dict

    def test_float16_output_is_cast_to_float32(self) -> None:
        m = ConcreteOnnxTextModel()
        session = MagicMock()
        session.run.return_value = [np.ones((1, 4), dtype=np.float16)]
        node = MagicMock()
        node.name = "input_ids"
        session.get_inputs.return_value = [node]
        m.model = session
        m.model_input_names = {"input_ids"}
        m.tokenizer = _make_mock_tokenizer()

        ctx = m.onnx_embed(["test"])
        assert ctx.model_output.dtype == np.float32

    def test_without_attention_mask_in_names_uses_raw_mask(self) -> None:
        m = ConcreteOnnxTextModel()
        session = MagicMock()
        session.run.return_value = [np.ones((1, 4), dtype=np.float32)]
        node = MagicMock()
        node.name = "input_ids"
        session.get_inputs.return_value = [node]
        m.model = session
        m.model_input_names = {"input_ids"}  # no attention_mask
        m.tokenizer = _make_mock_tokenizer()

        ctx = m.onnx_embed(["test"])
        assert ctx.attention_mask is not None

    def test_preprocess_onnx_input_is_called(self) -> None:
        """_preprocess_onnx_input (line 48) is reached through onnx_embed."""
        original = ConcreteOnnxTextModel._preprocess_onnx_input
        calls: list[dict] = []

        def spy(self, onnx_input, **kwargs):
            calls.append(onnx_input)
            return original(self, onnx_input, **kwargs)

        with patch.object(ConcreteOnnxTextModel, "_preprocess_onnx_input", spy):
            self.m.onnx_embed(["hello"])

        assert len(calls) == 1


class TestOnnxTextModelLoadOnnxModel:
    """Lines 60-69 — _load_onnx_model calls super then load_tokenizer."""

    def test_wires_tokenizer_after_super(self, tmp_path: Path) -> None:
        m = ConcreteOnnxTextModel()
        mock_tokenizer = MagicMock()
        mock_special: dict[str, int] = {"<s>": 1}

        with (
            patch("qwen3_embed.common.onnx_model.ort") as mock_ort,
            patch(
                "qwen3_embed.common.onnx_model.load_tokenizer",
                return_value=(mock_tokenizer, mock_special),
            ),
        ):
            mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]
            so_mock = MagicMock()
            mock_ort.SessionOptions.return_value = so_mock
            mock_ort.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
            session_mock = MagicMock()
            session_mock.get_providers.return_value = ["CPUExecutionProvider"]
            session_mock.get_inputs.return_value = []
            mock_ort.InferenceSession.return_value = session_mock

            m._load_onnx_model(tmp_path, "model.onnx", config=OnnxSessionConfig(threads=None))

        assert m.tokenizer is mock_tokenizer
        assert m.special_token_to_id is mock_special


class TestOnnxTextModelTokenCount:
    """Lines 168-178 — _token_count with lazy loading and various input forms."""

    def test_loads_model_if_not_loaded(self) -> None:
        m = ConcreteOnnxTextModel()
        m.model = None
        tok = _make_mock_tokenizer()
        enc = tok.encode_batch.return_value[0]
        enc.attention_mask = [1, 1, 0, 0]
        loaded: list[bool] = []

        def _fake_load() -> None:
            m.model = MagicMock()
            m.tokenizer = tok
            loaded.append(True)

        m.load_onnx_model = _fake_load  # type: ignore
        count = m._token_count(["hello"])
        assert loaded
        assert count == 2

    def test_single_string_wrapped_to_list(self) -> None:
        m = ConcreteOnnxTextModel()
        m.model = MagicMock()
        tok = _make_mock_tokenizer()
        enc = tok.encode_batch.return_value[0]
        enc.attention_mask = [1, 1, 1, 0]
        m.tokenizer = tok
        assert m._token_count("hello") == 3

    def test_multiple_texts_sum(self) -> None:
        m = ConcreteOnnxTextModel()
        m.model = MagicMock()
        enc1 = MagicMock()
        enc1.attention_mask = [1, 1, 0]  # 2 tokens
        enc2 = MagicMock()
        enc2.attention_mask = [1, 1, 1]  # 3 tokens
        tok = MagicMock()
        tok.encode_batch.return_value = [enc1, enc2]
        m.tokenizer = tok
        assert m._token_count(["foo", "bar"]) == 5


class TestOnnxTextModelEmbedDocuments:
    """Lines 124-165 — _embed_documents in single-process and parallel modes."""

    def _loaded(self, n_docs: int = 1) -> ConcreteOnnxTextModel:
        m = ConcreteOnnxTextModel()
        m.model = _make_mock_session((n_docs, 4))
        m.model_input_names = {"input_ids", "attention_mask"}
        m.tokenizer = _make_mock_tokenizer(n_docs=n_docs)
        return m

    def test_single_string_becomes_one_result(self) -> None:
        m = self._loaded(1)
        results = list(m._embed_documents("t", "/tmp", documents="hello"))
        assert len(results) == 1

    def test_small_list_all_results(self) -> None:
        # _post_process_onnx_output yields the whole batch array as one item,
        # so one batch of 2 docs produces 1 result (a (2, 4) array).
        m = self._loaded(2)
        results = list(m._embed_documents("t", "/tmp", documents=["a", "b"]))
        assert len(results) == 1
        assert results[0].shape == (2, 4)

    def test_loads_model_if_none(self) -> None:
        m = ConcreteOnnxTextModel()
        loaded: list[bool] = []

        def fake_load() -> None:
            m.model = _make_mock_session((1, 4))
            m.model_input_names = {"input_ids", "attention_mask"}
            m.tokenizer = _make_mock_tokenizer()
            loaded.append(True)

        m.load_onnx_model = fake_load  # type: ignore
        list(m._embed_documents("t", "/tmp", documents=["hi"]))
        assert loaded

    def _parallel_docs(self) -> list[str]:
        return ["a", "b", "c", "d", "e"]  # 5 docs, batch_size=2 => is_small=False

    def test_parallel_zero_creates_pool(self) -> None:
        m = self._loaded()
        out = OnnxOutputContext(model_output=np.ones((1, 4), dtype=np.float32))
        with patch("qwen3_embed.text.onnx_text_model.ParallelWorkerPool") as mock_cls:
            pool = MagicMock()
            pool.ordered_map.return_value = [out]
            mock_cls.return_value = pool
            list(
                m._embed_documents(
                    "t", "/tmp", documents=self._parallel_docs(), batch_size=2, parallel=0
                )
            )
        mock_cls.assert_called_once()

    def test_parallel_positive_sets_num_workers(self) -> None:
        m = self._loaded()
        out = OnnxOutputContext(model_output=np.ones((1, 4), dtype=np.float32))
        with patch("qwen3_embed.text.onnx_text_model.ParallelWorkerPool") as mock_cls:
            pool = MagicMock()
            pool.ordered_map.return_value = [out]
            mock_cls.return_value = pool
            list(
                m._embed_documents(
                    "t", "/tmp", documents=self._parallel_docs(), batch_size=2, parallel=3
                )
            )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["config"].num_workers == 3

    def test_parallel_extra_session_options_merged(self) -> None:
        m = self._loaded()
        out = OnnxOutputContext(model_output=np.ones((1, 4), dtype=np.float32))
        with patch("qwen3_embed.text.onnx_text_model.ParallelWorkerPool") as mock_cls:
            pool = MagicMock()
            pool.ordered_map.return_value = [out]
            mock_cls.return_value = pool
            list(
                m._embed_documents(
                    "t",
                    "/tmp",
                    documents=self._parallel_docs(),
                    batch_size=2,
                    parallel=2,
                    extra_session_options={"enable_cpu_mem_arena": False},
                )
            )
        mock_cls.assert_called_once()

    def test_parallel_uses_forkserver_or_spawn(self) -> None:
        """start_method selection doesn't crash."""
        m = self._loaded()
        out = OnnxOutputContext(model_output=np.ones((1, 4), dtype=np.float32))
        with patch("qwen3_embed.text.onnx_text_model.ParallelWorkerPool") as mock_cls:
            pool = MagicMock()
            pool.ordered_map.return_value = [out]
            mock_cls.return_value = pool
            list(
                m._embed_documents(
                    "t", "/tmp", documents=self._parallel_docs(), batch_size=2, parallel=1
                )
            )
        call_kwargs = mock_cls.call_args[1]
        assert call_kwargs["config"].start_method in ("forkserver", "spawn")


class TestOnnxTextModelStaticBatchGraph:
    """A graph pinned to a fixed batch must never be fed a larger one.

    Feeding it more rows aborts inside onnxruntime with a buffer shape
    mismatch, so the requested batch size is clamped to what the graph
    declares -- and left alone when the batch axis is dynamic.
    """

    def _session(self, batch_dim: int | str) -> MagicMock:
        """Mock session shaped like a real text-embedding graph."""
        session = MagicMock()
        session.run.return_value = [np.ones((1, 4), dtype=np.float32)]
        node_ids = MagicMock()
        node_ids.name = "input_ids"
        node_ids.shape = ["batch_size", "sequence_length"]
        session.get_inputs.return_value = [node_ids]
        node_out = MagicMock()
        node_out.name = "last_hidden_state"
        node_out.shape = [batch_dim, "sequence_length", 4]
        session.get_outputs.return_value = [node_out]
        return session

    def _tokenizer(self) -> MagicMock:
        """Tokenizer whose output length tracks the batch it is given."""
        enc = MagicMock()
        enc.ids = [1, 2, 3, 0]
        enc.attention_mask = [1, 1, 1, 0]
        tok = MagicMock()
        tok.encode_batch.side_effect = lambda documents: [enc] * len(documents)
        return tok

    def _model(self, batch_dim: int | str) -> tuple[ConcreteOnnxTextModel, MagicMock]:
        session = self._session(batch_dim)
        m = ConcreteOnnxTextModel()
        m.model = session
        m.model_input_names = {"input_ids"}
        m.tokenizer = self._tokenizer()
        m.static_batch_size = m._detect_static_batch_size(session)
        return m, session

    @staticmethod
    def _batch_widths(session: MagicMock) -> list[int]:
        return [call[0][1]["input_ids"].shape[0] for call in session.run.call_args_list]

    def test_static_batch_graph_is_fed_one_document_at_a_time(self) -> None:
        m, session = self._model(batch_dim=1)
        documents = [f"document {i}" for i in range(8)]

        list(m._embed_documents("t", "/tmp", documents=documents))

        assert self._batch_widths(session) == [1] * 8

    def test_dynamic_batch_graph_keeps_the_requested_batch(self) -> None:
        m, session = self._model(batch_dim="batch_size")
        documents = [f"document {i}" for i in range(8)]

        list(m._embed_documents("t", "/tmp", documents=documents))

        assert self._batch_widths(session) == [8]

    def test_caller_batch_size_below_the_graph_limit_is_respected(self) -> None:
        m, session = self._model(batch_dim=4)
        documents = [f"document {i}" for i in range(6)]

        list(m._embed_documents("t", "/tmp", documents=documents, batch_size=2))

        assert self._batch_widths(session) == [2, 2, 2]

    def test_unloaded_session_leaves_the_batch_size_alone(self) -> None:
        """With lazy_load plus parallel the graph lives only in the workers."""
        m = ConcreteOnnxTextModel()
        assert m._effective_batch_size(256) == 256


class TestTextEmbeddingWorkerProcess:
    """Lines 183-185 — TextEmbeddingWorker.process yields (idx, OnnxOutputContext)."""

    def test_yields_indexed_outputs(self) -> None:
        worker = _StubWorker(model_name="t", cache_dir="/tmp")
        items = [(0, ["hello"]), (1, ["world"])]
        results = list(worker.process(items))
        assert len(results) == 2
        assert results[0][0] == 0
        assert isinstance(results[0][1], OnnxOutputContext)
        assert results[1][0] == 1


# ===========================================================================
# OnnxTextEmbedding — covers onnx_embedding.py
# ===========================================================================


class TestOnnxTextEmbeddingInit:
    """Lines 65-91 — OnnxTextEmbedding.__init__ behaviour."""

    def test_lazy_load_skips_session(
        self, tmp_path: Path, registered_test_model: DenseModelDescription
    ) -> None:
        with patch.object(OnnxTextEmbedding, "download_model", return_value=tmp_path):
            emb = OnnxTextEmbedding(model_name=_MODEL_NAME, lazy_load=True)
        assert emb.lazy_load is True
        # Due to MRO: TextEmbeddingBase.__init__ does not call super().__init__(),
        # so OnnxModel.__init__ is never invoked and self.model is never set.
        assert not hasattr(emb, "model") or emb.model is None

    def test_lazy_load_false_calls_load_onnx_model(
        self, tmp_path: Path, registered_test_model: DenseModelDescription
    ) -> None:
        called: list[bool] = []
        with (
            patch.object(OnnxTextEmbedding, "download_model", return_value=tmp_path),
            patch.object(
                OnnxTextEmbedding, "load_onnx_model", side_effect=lambda: called.append(True)
            ),
        ):
            OnnxTextEmbedding(model_name=_MODEL_NAME, lazy_load=False)
        assert called

    def test_device_id_set_directly(
        self, tmp_path: Path, registered_test_model: DenseModelDescription
    ) -> None:
        with patch.object(OnnxTextEmbedding, "download_model", return_value=tmp_path):
            emb = OnnxTextEmbedding(model_name=_MODEL_NAME, lazy_load=True, device_id=7)
        assert emb.device_id == 7

    def test_device_id_from_device_ids(
        self, tmp_path: Path, registered_test_model: DenseModelDescription
    ) -> None:
        with patch.object(OnnxTextEmbedding, "download_model", return_value=tmp_path):
            emb = OnnxTextEmbedding(model_name=_MODEL_NAME, lazy_load=True, device_ids=[2, 3])
        assert emb.device_id == 2

    def test_no_device_id_defaults_to_none(
        self, tmp_path: Path, registered_test_model: DenseModelDescription
    ) -> None:
        with patch.object(OnnxTextEmbedding, "download_model", return_value=tmp_path):
            emb = OnnxTextEmbedding(model_name=_MODEL_NAME, lazy_load=True)
        assert emb.device_id is None

    def test_specific_model_path_stored(
        self, tmp_path: Path, registered_test_model: DenseModelDescription
    ) -> None:
        with patch.object(OnnxTextEmbedding, "download_model", return_value=tmp_path):
            emb = OnnxTextEmbedding(
                model_name=_MODEL_NAME,
                lazy_load=True,
                specific_model_path="/custom",
            )
        assert emb._specific_model_path == "/custom"


@pytest.fixture()
def onnx_emb(tmp_path: Path, registered_test_model: DenseModelDescription) -> OnnxTextEmbedding:
    """OnnxTextEmbedding with lazy_load + pre-wired mock session and tokenizer."""
    with patch.object(OnnxTextEmbedding, "download_model", return_value=tmp_path):
        emb = OnnxTextEmbedding(model_name=_MODEL_NAME, lazy_load=True)

    session = _make_mock_session((1, 4))
    tok = _make_mock_tokenizer(n_docs=1)
    emb.model = session
    emb.model_input_names = {"input_ids", "attention_mask"}
    emb.tokenizer = tok
    return emb


class TestOnnxTextEmbeddingMethods:
    def test_preprocess_returns_unchanged(self, onnx_emb: OnnxTextEmbedding) -> None:
        """Line 140."""
        data: dict[str, NumpyArray] = {"input_ids": np.array([[1, 2, 3]])}
        assert onnx_emb._preprocess_onnx_input(data) is data

    def test_get_worker_class(self, onnx_emb: OnnxTextEmbedding) -> None:
        """Line 132."""
        assert onnx_emb._get_worker_class() is OnnxTextEmbeddingWorker

    def test_post_process_2d_output(self, onnx_emb: OnnxTextEmbedding) -> None:
        """Line 150, 153 — 2D embeddings are normalized row-wise."""
        output = OnnxOutputContext(model_output=np.ones((2, 4), dtype=np.float32))
        results = list(onnx_emb._post_process_onnx_output(output))
        assert len(results) == 2
        assert results[0].shape == (4,)

    def test_post_process_3d_output_takes_cls(self, onnx_emb: OnnxTextEmbedding) -> None:
        """Line 148, 153 — 3D embeddings use CLS (first) token."""
        output = OnnxOutputContext(model_output=np.ones((2, 5, 4), dtype=np.float32))
        results = list(onnx_emb._post_process_onnx_output(output))
        assert len(results) == 2
        assert results[0].shape == (4,)

    def test_post_process_unsupported_ndim_raises(self, onnx_emb: OnnxTextEmbedding) -> None:
        output = OnnxOutputContext(model_output=np.ones((1, 2, 3, 4), dtype=np.float32))
        with pytest.raises(ValueError, match="Unsupported embedding shape"):
            list(onnx_emb._post_process_onnx_output(output))

    def test_embed_yields_normalized_embeddings(self, onnx_emb: OnnxTextEmbedding) -> None:
        """Line 115."""
        results = list(onnx_emb.embed(["hello"]))
        assert len(results) == 1
        assert results[0].shape == (4,)
        np.testing.assert_allclose(np.linalg.norm(results[0]), 1.0, atol=1e-6)

    def test_token_count_sums_mask(self, onnx_emb: OnnxTextEmbedding) -> None:
        """Line 169."""
        enc = onnx_emb.tokenizer.encode_batch.return_value[0]  # type: ignore
        enc.attention_mask = [1, 1, 0, 0]
        assert onnx_emb.token_count("hello") == 2

    def test_load_onnx_model_delegates_to_load_onnx_model(
        self, tmp_path: Path, registered_test_model: DenseModelDescription
    ) -> None:
        """Line 156 — load_onnx_model calls _load_onnx_model with correct args."""
        with patch.object(OnnxTextEmbedding, "download_model", return_value=tmp_path):
            emb = OnnxTextEmbedding(model_name=_MODEL_NAME, lazy_load=True)

        with patch.object(emb, "_load_onnx_model") as mock_load:
            emb.load_onnx_model()

        mock_load.assert_called_once_with(
            model_dir=tmp_path,
            model_file=_MODEL_DESC.model_file,
            config=OnnxSessionConfig(
                threads=emb.threads,
                providers=emb.providers,
                cuda=emb.cuda,
                device_id=emb.device_id,
                extra_session_options=emb._extra_session_options,
            ),
        )


class TestOnnxTextEmbeddingWorkerInit:
    """Line 179 — OnnxTextEmbeddingWorker.init_embedding creates OnnxTextEmbedding."""

    def test_creates_onnx_text_embedding(
        self, tmp_path: Path, registered_test_model: DenseModelDescription
    ) -> None:
        with (
            patch.object(OnnxTextEmbedding, "download_model", return_value=tmp_path),
            patch.object(OnnxTextEmbedding, "load_onnx_model"),
        ):
            worker = OnnxTextEmbeddingWorker(
                model_name=_MODEL_NAME,
                cache_dir=str(tmp_path),
            )
        assert isinstance(worker.model, OnnxTextEmbedding)
        assert worker.model.model_name == _MODEL_NAME


# ===========================================================================
# Robustness tests for OnnxTextModel
# ===========================================================================


class TestOnnxTextModelRobustness:
    def test_token_count_empty_generator(self) -> None:
        m = ConcreteOnnxTextModel()
        m.model = MagicMock()
        m.tokenizer = _make_mock_tokenizer()  # type: ignore[invalid-assignment]

        def empty_gen():
            yield from []

        assert m._token_count(empty_gen()) == 0

    def test_onnx_embed_empty_list_raises_or_empty(self) -> None:
        m = ConcreteOnnxTextModel()
        m.model = _make_mock_session((0, 4))
        m.model_input_names = {"input_ids"}
        tok = MagicMock()
        tok.encode_batch.return_value = []
        m.tokenizer = tok  # type: ignore[invalid-assignment]

        ctx = m.onnx_embed([])
        assert ctx.model_output.shape[0] == 0

    def test_embed_documents_parallel_small_input_stays_single_process(self) -> None:
        m = ConcreteOnnxTextModel()
        m.model = _make_mock_session((1, 4))
        m.model_input_names = {"input_ids"}
        m.tokenizer = _make_mock_tokenizer()  # type: ignore[invalid-assignment]

        # parallel=2 but only 1 document, batch_size=256
        with patch("qwen3_embed.text.onnx_text_model.ParallelWorkerPool") as mock_pool:
            results = list(m._embed_documents("t", "/tmp", documents=["hi"], parallel=2))
            assert len(results) == 1
            mock_pool.assert_not_called()

    def test_extra_worker_params_propagation(self) -> None:
        class CustomModel(ConcreteOnnxTextModel):
            def _extra_worker_params(self):
                return {"custom_key": "custom_val"}

        m = CustomModel()
        m.model = _make_mock_session()
        m.tokenizer = _make_mock_tokenizer()  # type: ignore[invalid-assignment]

        out = OnnxOutputContext(model_output=np.ones((1, 4)))
        with patch("qwen3_embed.text.onnx_text_model.ParallelWorkerPool") as mock_cls:
            pool = MagicMock()
            pool.ordered_map.return_value = [out]
            mock_cls.return_value = pool

            # documents length must be >= batch_size to trigger parallel if parallel is not small
            list(m._embed_documents("t", "/tmp", documents=["a", "b"], batch_size=1, parallel=2))

        # Verify params passed to ordered_map or constructor?
        # Params are merged into 'params' dict passed to pool.ordered_map
        _, kwargs = pool.ordered_map.call_args
        assert kwargs["custom_key"] == "custom_val"
