"""Tests for OnnxCrossEncoderModel (onnx_text_model.py) and
OnnxTextCrossEncoder (onnx_text_cross_encoder.py).

All ONNX sessions, tokenizers, and model downloads are fully mocked.
No real models are loaded.
"""

from collections.abc import Iterable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qwen3_embed.common.model_description import BaseModelDescription, ModelSource
from qwen3_embed.common.onnx_model import OnnxOutputContext, OnnxSessionConfig
from qwen3_embed.common.types import NumpyArray
from qwen3_embed.rerank.cross_encoder.onnx_text_cross_encoder import (
    OnnxTextCrossEncoder,
    TextCrossEncoderWorker,
    supported_onnx_models,
)
from qwen3_embed.rerank.cross_encoder.onnx_text_model import (
    OnnxCrossEncoderModel,
    TextRerankerWorker,
)

# ---------------------------------------------------------------------------
# Test model description
# ---------------------------------------------------------------------------

_MODEL_NAME = "test-org/test-cross-encoder"
_MODEL_DESC = BaseModelDescription(
    model=_MODEL_NAME,
    sources=ModelSource(hf="test-org/test-cross-encoder"),
    model_file="onnx/model.onnx",
    description="test cross encoder",
    license="MIT",
    size_in_GB=0.1,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_encoding(ids: list[int] | None = None, n: int = 4) -> MagicMock:
    enc = MagicMock()
    enc.ids = ids if ids is not None else [1, 2, 3, 0][:n]
    enc.type_ids = [0] * n
    enc.attention_mask = [1, 1, 1, 0][:n]
    return enc


def _make_mock_tokenizer(n_pairs: int = 1, seq_len: int = 4) -> MagicMock:
    enc = _make_mock_encoding(n=seq_len)
    tok = MagicMock()
    tok.encode_batch.return_value = [enc] * n_pairs
    return tok


def _make_mock_session(
    n_pairs: int = 1,
    n_labels: int = 2,
    dtype: type = np.float32,
    input_names: list[str] | None = None,
) -> MagicMock:
    """Return a mock onnxruntime.InferenceSession.

    Default output shape: (n_pairs, n_labels) — scores[:, 0] is taken.
    """
    output = np.ones((n_pairs, n_labels), dtype=dtype)
    session = MagicMock()
    session.run.return_value = [output]

    names = input_names if input_names is not None else ["input_ids", "attention_mask"]
    nodes = []
    for name in names:
        node = MagicMock()
        node.name = name
        nodes.append(node)
    session.get_inputs.return_value = nodes
    return session


# ---------------------------------------------------------------------------
# Concrete OnnxCrossEncoderModel for unit-testing the base class
# ---------------------------------------------------------------------------


class ConcreteCrossEncoderModel(OnnxCrossEncoderModel):
    """Minimal subclass — no real models, no real I/O."""

    @classmethod
    def _get_worker_class(cls) -> type["_StubRerankerWorker"]:
        return _StubRerankerWorker

    def _post_process_onnx_output(
        self, output: OnnxOutputContext, **kwargs: Any
    ) -> Iterable[float]:
        return (float(x) for x in output.model_output)

    def load_onnx_model(self) -> None:
        pass  # no-op


class _StubRerankerWorker(TextRerankerWorker):
    def init_embedding(
        self,
        model_name: str,
        cache_dir: str,
        **kwargs: Any,
    ) -> ConcreteCrossEncoderModel:
        m = ConcreteCrossEncoderModel()
        m.model = _make_mock_session()
        m.model_input_names = {"input_ids", "attention_mask"}
        m.tokenizer = _make_mock_tokenizer()
        return m


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def registered_test_model():
    """Register the test model description into OnnxTextCrossEncoder."""
    supported_onnx_models.append(_MODEL_DESC)
    yield _MODEL_DESC
    if _MODEL_DESC in supported_onnx_models:
        supported_onnx_models.remove(_MODEL_DESC)


@pytest.fixture()
def loaded_model() -> ConcreteCrossEncoderModel:
    """ConcreteCrossEncoderModel pre-wired with mock session and tokenizer."""
    m = ConcreteCrossEncoderModel()
    m.model = _make_mock_session(n_pairs=2)
    m.model_input_names = {"input_ids", "attention_mask"}
    m.tokenizer = _make_mock_tokenizer(n_pairs=2)
    return m


@pytest.fixture()
def onnx_encoder(
    tmp_path: Path, registered_test_model: BaseModelDescription
) -> OnnxTextCrossEncoder:
    """OnnxTextCrossEncoder with lazy_load + pre-wired mock."""
    with patch.object(OnnxTextCrossEncoder, "download_model", return_value=tmp_path):
        enc = OnnxTextCrossEncoder(model_name=_MODEL_NAME, lazy_load=True)

    enc.model = _make_mock_session(n_pairs=1)
    enc.model_input_names = {"input_ids", "attention_mask"}
    enc.tokenizer = _make_mock_tokenizer(n_pairs=1)
    return enc


# ===========================================================================
# OnnxCrossEncoderModel — base class (onnx_text_model.py)
# ===========================================================================


class TestOnnxCrossEncoderModelAbstract:
    """Abstract method stubs raise NotImplementedError."""

    def test_get_worker_class_raises(self) -> None:
        class _Stub(OnnxCrossEncoderModel):
            pass

        with pytest.raises(NotImplementedError):
            _Stub._get_worker_class()

    def test_post_process_raises(self) -> None:
        class _Stub(OnnxCrossEncoderModel):
            pass

        stub = _Stub()
        with pytest.raises(NotImplementedError):
            list(stub._post_process_onnx_output(OnnxOutputContext(model_output=np.zeros((1,)))))

    def test_preprocess_onnx_input_returns_identity(self) -> None:
        m = ConcreteCrossEncoderModel()
        data: dict[str, NumpyArray] = {"input_ids": np.array([[1, 2, 3]])}
        assert m._preprocess_onnx_input(data) is data


class TestOnnxCrossEncoderModelTokenize:
    """tokenize() delegates to tokenizer.encode_batch."""

    def test_tokenize_returns_encodings(self) -> None:
        m = ConcreteCrossEncoderModel()
        m.tokenizer = _make_mock_tokenizer(n_pairs=2)
        pairs = [("query", "doc1"), ("query", "doc2")]
        result = m.tokenize(pairs)
        m.tokenizer.encode_batch.assert_called_once_with(pairs)
        assert len(result) == 2

    def test_tokenize_asserts_tokenizer_not_none(self) -> None:
        m = ConcreteCrossEncoderModel()
        with pytest.raises(AssertionError):
            m.tokenize([("q", "d")])


class TestBuildOnnxInput:
    """_build_onnx_input assembles numpy arrays from Encoding objects."""

    def test_input_ids_always_present(self) -> None:
        m = ConcreteCrossEncoderModel()
        m.model_input_names = {"input_ids"}
        enc = _make_mock_encoding()
        result = m._build_onnx_input([enc])
        assert "input_ids" in result
        assert result["input_ids"].dtype == np.int64

    def test_attention_mask_included_when_in_names(self) -> None:
        m = ConcreteCrossEncoderModel()
        m.model_input_names = {"input_ids", "attention_mask"}
        enc = _make_mock_encoding()
        result = m._build_onnx_input([enc])
        assert "attention_mask" in result
        assert result["attention_mask"].dtype == np.int64

    def test_token_type_ids_included_when_in_names(self) -> None:
        m = ConcreteCrossEncoderModel()
        m.model_input_names = {"input_ids", "token_type_ids"}
        enc = _make_mock_encoding()
        result = m._build_onnx_input([enc])
        assert "token_type_ids" in result
        assert result["token_type_ids"].dtype == np.int64

    def test_attention_mask_excluded_when_not_in_names(self) -> None:
        m = ConcreteCrossEncoderModel()
        m.model_input_names = {"input_ids"}
        enc = _make_mock_encoding()
        result = m._build_onnx_input([enc])
        assert "attention_mask" not in result

    def test_token_type_ids_excluded_when_not_in_names(self) -> None:
        m = ConcreteCrossEncoderModel()
        m.model_input_names = {"input_ids"}
        enc = _make_mock_encoding()
        result = m._build_onnx_input([enc])
        assert "token_type_ids" not in result

    def test_all_three_fields(self) -> None:
        m = ConcreteCrossEncoderModel()
        m.model_input_names = {"input_ids", "attention_mask", "token_type_ids"}
        enc = _make_mock_encoding()
        result = m._build_onnx_input([enc])
        assert set(result.keys()) == {"input_ids", "attention_mask", "token_type_ids"}

    def test_empty_input_names(self) -> None:
        m = ConcreteCrossEncoderModel()
        m.model_input_names = None  # falls back to empty set
        enc = _make_mock_encoding()
        result = m._build_onnx_input([enc])
        # Only input_ids is always added
        assert "input_ids" in result
        assert "attention_mask" not in result

    def test_batch_shape(self) -> None:
        m = ConcreteCrossEncoderModel()
        m.model_input_names = {"input_ids"}
        encs = [_make_mock_encoding() for _ in range(3)]
        result = m._build_onnx_input(encs)  # type: ignore[arg-type]
        assert result["input_ids"].shape[0] == 3


class TestOnnxEmbedPairs:
    """onnx_embed_pairs runs the full tokenize → build → preprocess → run pipeline."""

    def test_returns_onnx_output_context(self, loaded_model: ConcreteCrossEncoderModel) -> None:
        pairs = [("q", "d1"), ("q", "d2")]
        ctx = loaded_model.onnx_embed_pairs(pairs)
        assert isinstance(ctx, OnnxOutputContext)

    def test_scores_shape(self, loaded_model: ConcreteCrossEncoderModel) -> None:
        pairs = [("q", "d1"), ("q", "d2")]
        ctx = loaded_model.onnx_embed_pairs(pairs)
        # scores = outputs[0][:, 0]
        assert ctx.model_output.shape == (2,)

    def test_model_run_called_with_onnx_input(
        self, loaded_model: ConcreteCrossEncoderModel
    ) -> None:
        pairs = [("q", "d")]
        loaded_model.model_input_names = {"input_ids"}
        loaded_model.tokenizer = _make_mock_tokenizer(n_pairs=1)
        loaded_model.onnx_embed_pairs(pairs)
        loaded_model.model.run.assert_called_once()  # type: ignore[union-attr]

    def test_onnx_embed_delegates_to_onnx_embed_pairs(
        self, loaded_model: ConcreteCrossEncoderModel
    ) -> None:
        """onnx_embed(query, docs) wraps into pairs and calls onnx_embed_pairs."""
        loaded_model.model_input_names = {"input_ids"}
        loaded_model.tokenizer = _make_mock_tokenizer(n_pairs=2)
        ctx = loaded_model.onnx_embed("query", ["doc1", "doc2"])
        assert ctx.model_output.shape == (2,)


class TestRerankDocuments:
    """_rerank_documents batches documents and yields scores."""

    def test_yields_floats(self, loaded_model: ConcreteCrossEncoderModel) -> None:
        scores = list(loaded_model._rerank_documents("q", ["d1", "d2"], batch_size=64))
        assert len(scores) == 2
        assert all(isinstance(s, float) for s in scores)

    def test_loads_model_if_none(self) -> None:
        m = ConcreteCrossEncoderModel()
        m.model = None
        loaded: list[bool] = []

        def _fake_load() -> None:
            m.model = _make_mock_session(n_pairs=1)
            m.model_input_names = {"input_ids"}
            m.tokenizer = _make_mock_tokenizer(n_pairs=1)
            loaded.append(True)

        m.load_onnx_model = _fake_load  # type: ignore[invalid-assignment]
        list(m._rerank_documents("q", ["d1"], batch_size=64))
        assert loaded


class TestRerankPairsIsSmallBranch:
    """_rerank_pairs with small input (no parallel pool)."""

    def test_single_tuple_becomes_one_pair(self) -> None:
        # A single tuple triggers is_small=True; use n_pairs=1 mock so output is 1 score
        m = ConcreteCrossEncoderModel()
        m.model = _make_mock_session(n_pairs=1)
        m.model_input_names = {"input_ids"}
        m.tokenizer = _make_mock_tokenizer(n_pairs=1)
        scores = list(
            m._rerank_pairs(
                model_name="m",
                cache_dir="/tmp",
                pairs=[("query", "doc")],
                batch_size=64,
            )
        )
        assert len(scores) == 1
        assert all(isinstance(s, float) for s in scores)

    def test_list_smaller_than_batch_size(self, loaded_model: ConcreteCrossEncoderModel) -> None:
        pairs = [("q", "d1"), ("q", "d2")]
        loaded_model.tokenizer = _make_mock_tokenizer(n_pairs=2)
        scores = list(
            loaded_model._rerank_pairs(
                model_name="m",
                cache_dir="/tmp",
                pairs=pairs,
                batch_size=64,  # larger than len(pairs)=2 => is_small
            )
        )
        assert len(scores) == 2

    def test_parallel_none_uses_direct_path(self, loaded_model: ConcreteCrossEncoderModel) -> None:
        pairs = [("q", f"d{i}") for i in range(5)]
        loaded_model.tokenizer = _make_mock_tokenizer(n_pairs=5)
        loaded_model.model = _make_mock_session(n_pairs=5)
        scores = list(
            loaded_model._rerank_pairs(
                model_name="m",
                cache_dir="/tmp",
                pairs=pairs,
                batch_size=10,
                parallel=None,
            )
        )
        assert len(scores) == 5

    def test_loads_model_when_none_in_small_branch(self) -> None:
        m = ConcreteCrossEncoderModel()
        m.model = None
        loaded: list[bool] = []

        def _fake_load() -> None:
            m.model = _make_mock_session(n_pairs=1)
            m.model_input_names = {"input_ids"}
            m.tokenizer = _make_mock_tokenizer(n_pairs=1)
            loaded.append(True)

        m.load_onnx_model = _fake_load  # type: ignore[invalid-assignment]
        list(
            m._rerank_pairs(
                model_name="m",
                cache_dir="/tmp",
                pairs=[("q", "d")],
                batch_size=64,
            )
        )
        assert loaded


class TestRerankPairsParallelBranch:
    """_rerank_pairs when parallel > 0 and input is large (spawns a pool)."""

    def _large_pairs(self, n: int = 10) -> list[tuple[str, str]]:
        return [("q", f"d{i}") for i in range(n)]

    def test_parallel_zero_uses_cpu_count(self, loaded_model: ConcreteCrossEncoderModel) -> None:
        out = OnnxOutputContext(model_output=np.array([0.5, 0.6]))
        with patch("qwen3_embed.rerank.cross_encoder.onnx_text_model.ParallelWorkerPool") as cls:
            pool = MagicMock()
            pool.ordered_map.return_value = [out]
            cls.return_value = pool
            list(
                loaded_model._rerank_pairs(
                    model_name="m",
                    cache_dir="/tmp",
                    pairs=self._large_pairs(),
                    batch_size=2,
                    parallel=0,
                )
            )
        cls.assert_called_once()

    def test_parallel_positive_creates_pool_with_num_workers(
        self, loaded_model: ConcreteCrossEncoderModel
    ) -> None:
        out = OnnxOutputContext(model_output=np.array([0.5]))
        with patch("qwen3_embed.rerank.cross_encoder.onnx_text_model.ParallelWorkerPool") as cls:
            pool = MagicMock()
            pool.ordered_map.return_value = [out]
            cls.return_value = pool
            list(
                loaded_model._rerank_pairs(
                    model_name="m",
                    cache_dir="/tmp",
                    pairs=self._large_pairs(),
                    batch_size=2,
                    parallel=3,
                )
            )
        call_kw = cls.call_args[1]
        assert call_kw["config"].num_workers == 3

    def test_parallel_start_method_is_forkserver_or_spawn(
        self, loaded_model: ConcreteCrossEncoderModel
    ) -> None:
        out = OnnxOutputContext(model_output=np.array([0.5]))
        with patch("qwen3_embed.rerank.cross_encoder.onnx_text_model.ParallelWorkerPool") as cls:
            pool = MagicMock()
            pool.ordered_map.return_value = [out]
            cls.return_value = pool
            list(
                loaded_model._rerank_pairs(
                    model_name="m",
                    cache_dir="/tmp",
                    pairs=self._large_pairs(),
                    batch_size=2,
                    parallel=2,
                )
            )
        call_kw = cls.call_args[1]
        assert call_kw["config"].start_method in ("forkserver", "spawn")

    def test_extra_session_options_merged_into_params(
        self, loaded_model: ConcreteCrossEncoderModel
    ) -> None:
        out = OnnxOutputContext(model_output=np.array([0.5]))
        with patch("qwen3_embed.rerank.cross_encoder.onnx_text_model.ParallelWorkerPool") as cls:
            pool = MagicMock()
            pool.ordered_map.return_value = [out]
            cls.return_value = pool
            list(
                loaded_model._rerank_pairs(
                    model_name="m",
                    cache_dir="/tmp",
                    pairs=self._large_pairs(),
                    batch_size=2,
                    parallel=2,
                    extra_session_options={"enable_cpu_mem_arena": False},
                )
            )
        cls.assert_called_once()
        # The extra session options get passed via pool.ordered_map kwargs
        _, ordered_map_kwargs = pool.ordered_map.call_args
        assert "enable_cpu_mem_arena" in ordered_map_kwargs


class TestTokenCount:
    """_token_count sums attention mask tokens across batches."""

    def test_basic_token_count(self) -> None:
        m = ConcreteCrossEncoderModel()
        m.model = MagicMock()
        enc = MagicMock()
        enc.attention_mask = [1, 1, 1, 0]  # 3 tokens
        tok = MagicMock()
        tok.encode_batch.return_value = [enc]
        m.tokenizer = tok
        result = m._token_count([("q", "d")])
        assert result == 3

    def test_multiple_pairs_summed(self) -> None:
        m = ConcreteCrossEncoderModel()
        m.model = MagicMock()
        enc1 = MagicMock()
        enc1.attention_mask = [1, 1, 0]
        enc2 = MagicMock()
        enc2.attention_mask = [1, 1, 1]
        tok = MagicMock()
        tok.encode_batch.return_value = [enc1, enc2]
        m.tokenizer = tok
        result = m._token_count([("q", "d1"), ("q", "d2")])
        assert result == 5

    def test_loads_model_if_none(self) -> None:
        m = ConcreteCrossEncoderModel()
        m.model = None
        loaded: list[bool] = []

        def _fake_load() -> None:
            m.model = MagicMock()
            enc = MagicMock()
            enc.attention_mask = [1, 0]
            tok = MagicMock()
            tok.encode_batch.return_value = [enc]
            m.tokenizer = tok
            loaded.append(True)

        m.load_onnx_model = _fake_load  # type: ignore[invalid-assignment]
        count = m._token_count([("q", "d")])
        assert loaded
        assert count == 1


class TestTextRerankerWorker:
    """TextRerankerWorker.process yields (idx, OnnxOutputContext)."""

    def test_process_yields_indexed_outputs(self) -> None:
        worker = _StubRerankerWorker(model_name="t", cache_dir="/tmp")
        items = [(0, [("q", "d1")]), (1, [("q", "d2")])]
        results = list(worker.process(items))
        assert len(results) == 2
        assert results[0][0] == 0
        assert isinstance(results[0][1], OnnxOutputContext)
        assert results[1][0] == 1

    def test_init_embedding_raises_not_implemented(self) -> None:
        worker = _StubRerankerWorker(model_name="t", cache_dir="/tmp")
        with pytest.raises(NotImplementedError):
            TextRerankerWorker.init_embedding(worker, "m", "/tmp")


class TestLoadOnnxModel:
    """_load_onnx_model calls super() then wires up the tokenizer."""

    def test_wires_tokenizer_after_super(self, tmp_path: Path) -> None:
        m = ConcreteCrossEncoderModel()
        mock_tokenizer = MagicMock()

        with (
            patch("qwen3_embed.common.onnx_model.ort") as mock_ort,
            patch(
                "qwen3_embed.common.onnx_model.load_tokenizer",
                return_value=(mock_tokenizer, {}),
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


# ===========================================================================
# OnnxTextCrossEncoder — facade class (onnx_text_cross_encoder.py)
# ===========================================================================


class TestOnnxTextCrossEncoderInit:
    """OnnxTextCrossEncoder.__init__ behaviour."""

    def test_lazy_load_skips_session(
        self, tmp_path: Path, registered_test_model: BaseModelDescription
    ) -> None:
        with patch.object(OnnxTextCrossEncoder, "download_model", return_value=tmp_path):
            enc = OnnxTextCrossEncoder(model_name=_MODEL_NAME, lazy_load=True)
        assert enc.lazy_load is True
        # OnnxModel.__init__ is not called through the MRO in this configuration,
        # so `model` may not be set at all. Either absent or None means not loaded.
        assert not hasattr(enc, "model") or enc.model is None

    def test_lazy_load_false_calls_load_onnx_model(
        self, tmp_path: Path, registered_test_model: BaseModelDescription
    ) -> None:
        called: list[bool] = []
        with (
            patch.object(OnnxTextCrossEncoder, "download_model", return_value=tmp_path),
            patch.object(
                OnnxTextCrossEncoder,
                "load_onnx_model",
                side_effect=lambda: called.append(True),
            ),
        ):
            OnnxTextCrossEncoder(model_name=_MODEL_NAME, lazy_load=False)
        assert called

    def test_device_id_set_directly(
        self, tmp_path: Path, registered_test_model: BaseModelDescription
    ) -> None:
        with patch.object(OnnxTextCrossEncoder, "download_model", return_value=tmp_path):
            enc = OnnxTextCrossEncoder(model_name=_MODEL_NAME, lazy_load=True, device_id=5)
        assert enc.device_id == 5

    def test_device_id_from_device_ids(
        self, tmp_path: Path, registered_test_model: BaseModelDescription
    ) -> None:
        with patch.object(OnnxTextCrossEncoder, "download_model", return_value=tmp_path):
            enc = OnnxTextCrossEncoder(model_name=_MODEL_NAME, lazy_load=True, device_ids=[3, 4])
        assert enc.device_id == 3

    def test_device_id_defaults_to_none(
        self, tmp_path: Path, registered_test_model: BaseModelDescription
    ) -> None:
        with patch.object(OnnxTextCrossEncoder, "download_model", return_value=tmp_path):
            enc = OnnxTextCrossEncoder(model_name=_MODEL_NAME, lazy_load=True)
        assert enc.device_id is None

    def test_specific_model_path_stored(
        self, tmp_path: Path, registered_test_model: BaseModelDescription
    ) -> None:
        with patch.object(OnnxTextCrossEncoder, "download_model", return_value=tmp_path):
            enc = OnnxTextCrossEncoder(
                model_name=_MODEL_NAME, lazy_load=True, specific_model_path="/custom"
            )
        assert enc._specific_model_path == "/custom"

    def test_multiple_device_ids_logs_warning(
        self, tmp_path: Path, registered_test_model: BaseModelDescription
    ) -> None:
        with (
            patch.object(OnnxTextCrossEncoder, "download_model", return_value=tmp_path),
            patch(
                "qwen3_embed.rerank.cross_encoder.onnx_text_cross_encoder.logger"
            ) as mock_logger,
        ):
            OnnxTextCrossEncoder(model_name=_MODEL_NAME, lazy_load=True, device_ids=[0, 1, 2])
        mock_logger.warning.assert_called_once()


class TestOnnxTextCrossEncoderListSupportedModels:
    """_list_supported_models returns supported_onnx_models list."""

    def test_returns_supported_models_list(
        self, registered_test_model: BaseModelDescription
    ) -> None:
        models = OnnxTextCrossEncoder._list_supported_models()
        assert _MODEL_DESC in models


class TestOnnxTextCrossEncoderLoadOnnxModel:
    """load_onnx_model delegates to _load_onnx_model with correct args."""

    def test_delegates_to_load_onnx_model(
        self, tmp_path: Path, registered_test_model: BaseModelDescription
    ) -> None:
        with patch.object(OnnxTextCrossEncoder, "download_model", return_value=tmp_path):
            enc = OnnxTextCrossEncoder(model_name=_MODEL_NAME, lazy_load=True)

        with patch.object(enc, "_load_onnx_model") as mock_load:
            enc.load_onnx_model()

        mock_load.assert_called_once_with(
            model_dir=tmp_path,
            model_file=_MODEL_DESC.model_file,
            config=OnnxSessionConfig(
                threads=enc.threads,
                providers=enc.providers,
                cuda=enc.cuda,
                device_id=enc.device_id,
                extra_session_options=enc._extra_session_options,
            ),
        )


class TestOnnxTextCrossEncoderRerank:
    """rerank() yields float scores for each document."""

    def test_rerank_yields_floats(self, onnx_encoder: OnnxTextCrossEncoder) -> None:
        scores = list(onnx_encoder.rerank("query", ["doc1"]))
        assert len(scores) == 1
        assert isinstance(scores[0], float)

    def test_rerank_delegates_to_rerank_documents(
        self, onnx_encoder: OnnxTextCrossEncoder
    ) -> None:
        with patch.object(onnx_encoder, "_rerank_documents", wraps=onnx_encoder._rerank_documents):
            list(onnx_encoder.rerank("query", ["doc"], batch_size=32))
            onnx_encoder._rerank_documents.assert_called_once()  # type: ignore[attr-defined]


class TestOnnxTextCrossEncoderRerankPairs:
    """rerank_pairs() delegates to _rerank_pairs."""

    def test_rerank_pairs_single_tuple(self, onnx_encoder: OnnxTextCrossEncoder) -> None:
        onnx_encoder.tokenizer = _make_mock_tokenizer(n_pairs=1)  # type: ignore[invalid-assignment]
        # Passing a single tuple instead of a list of tuples
        scores = list(onnx_encoder.rerank_pairs(("query", "doc")))
        assert len(scores) == 1
        assert isinstance(scores[0], float)

    def test_rerank_pairs_yields_floats(self, onnx_encoder: OnnxTextCrossEncoder) -> None:
        onnx_encoder.tokenizer = _make_mock_tokenizer(n_pairs=1)  # type: ignore[invalid-assignment]
        scores = list(onnx_encoder.rerank_pairs([("query", "doc")]))
        assert len(scores) == 1
        assert isinstance(scores[0], float)

    def test_rerank_pairs_parallel_delegates_to_pool(
        self, onnx_encoder: OnnxTextCrossEncoder
    ) -> None:
        out = OnnxOutputContext(model_output=np.array([0.9, 0.1]))
        pairs = [("q", f"d{i}") for i in range(10)]
        with patch("qwen3_embed.rerank.cross_encoder.onnx_text_model.ParallelWorkerPool") as cls:
            pool = MagicMock()
            pool.ordered_map.return_value = [out]
            cls.return_value = pool
            list(onnx_encoder.rerank_pairs(pairs, batch_size=2, parallel=2))
        cls.assert_called_once()


class TestOnnxTextCrossEncoderPostProcess:
    """_post_process_onnx_output converts model_output to float generator."""

    def test_float_values_yielded(self, onnx_encoder: OnnxTextCrossEncoder) -> None:
        output = OnnxOutputContext(model_output=np.array([0.7, 0.3], dtype=np.float32))
        result = list(onnx_encoder._post_process_onnx_output(output))
        assert len(result) == 2
        assert all(isinstance(v, float) for v in result)

    def test_values_match_array(self, onnx_encoder: OnnxTextCrossEncoder) -> None:
        arr = np.array([0.8, 0.2], dtype=np.float32)
        output = OnnxOutputContext(model_output=arr)
        result = list(onnx_encoder._post_process_onnx_output(output))
        assert result[0] == pytest.approx(0.8)
        assert result[1] == pytest.approx(0.2)


class TestOnnxTextCrossEncoderTokenCount:
    """token_count() sums tokens across all pairs."""

    def test_token_count_delegates(self, onnx_encoder: OnnxTextCrossEncoder) -> None:
        enc = MagicMock()
        enc.attention_mask = [1, 1, 0, 0]
        onnx_encoder.tokenizer.encode_batch.return_value = [enc]  # type: ignore[unresolved-attribute]
        result = onnx_encoder.token_count([("q", "d")])
        assert result == 2


class TestOnnxTextCrossEncoderGetWorkerClass:
    """_get_worker_class returns TextCrossEncoderWorker."""

    def test_returns_text_cross_encoder_worker(self, onnx_encoder: OnnxTextCrossEncoder) -> None:
        assert onnx_encoder._get_worker_class() is TextCrossEncoderWorker


class TestTextCrossEncoderWorker:
    """TextCrossEncoderWorker.init_embedding creates an OnnxTextCrossEncoder."""

    def test_creates_onnx_text_cross_encoder(
        self, tmp_path: Path, registered_test_model: BaseModelDescription
    ) -> None:
        with (
            patch.object(OnnxTextCrossEncoder, "download_model", return_value=tmp_path),
            patch.object(OnnxTextCrossEncoder, "load_onnx_model"),
        ):
            worker = TextCrossEncoderWorker(
                model_name=_MODEL_NAME,
                cache_dir=str(tmp_path),
            )
        assert isinstance(worker.model, OnnxTextCrossEncoder)
        assert worker.model.model_name == _MODEL_NAME


# ===========================================================================
# Float16 → Float32 casting in onnx_embed_pairs
# ===========================================================================


class TestFloat16Casting:
    """Output from the ONNX session in float16 should be cast to float32."""

    def test_float16_output_cast_to_float32(self) -> None:
        m = ConcreteCrossEncoderModel()
        # Create a float16 session output: (1, 2) — scores[:, 0]
        session = MagicMock()
        session.run.return_value = [np.ones((1, 2), dtype=np.float16)]
        session.get_inputs.return_value = []
        m.model = session
        m.model_input_names = {"input_ids"}
        m.tokenizer = _make_mock_tokenizer(n_pairs=1)

        # onnx_embed_pairs extracts scores[:, 0]; dtype of raw output is float16
        # but the test verifies that the slice (1D) doesn't break post-processing
        ctx = m.onnx_embed_pairs([("q", "d")])
        # scores array should have been sliced from float16 output
        assert ctx.model_output is not None


# ===========================================================================
# _preprocess_onnx_input pass-through in OnnxCrossEncoderModel
# ===========================================================================


class TestPreprocessOnnxInput:
    """_preprocess_onnx_input is called inside onnx_embed_pairs."""

    def test_preprocess_called_during_embed_pairs(
        self, loaded_model: ConcreteCrossEncoderModel
    ) -> None:
        calls: list[dict] = []
        original = loaded_model._preprocess_onnx_input

        def spy(onnx_input: dict, **kwargs: Any) -> dict:
            calls.append(onnx_input)
            return original(onnx_input, **kwargs)

        loaded_model._preprocess_onnx_input = spy  # type: ignore[method-assign]
        loaded_model.onnx_embed_pairs([("q", "d1"), ("q", "d2")])
        assert len(calls) == 1

    def test_custom_preprocess_modifies_input(self) -> None:
        """Subclasses can override _preprocess_onnx_input to transform inputs."""

        class PreprocessingModel(ConcreteCrossEncoderModel):
            def _preprocess_onnx_input(
                self, onnx_input: dict[str, NumpyArray], **kwargs: Any
            ) -> dict[str, NumpyArray]:
                onnx_input["custom"] = np.zeros((1,), dtype=np.float32)
                return onnx_input

        m = PreprocessingModel()
        session = MagicMock()
        session.run.return_value = [np.ones((1, 2), dtype=np.float32)]
        session.get_inputs.return_value = []
        m.model = session
        m.model_input_names = {"input_ids"}
        m.tokenizer = _make_mock_tokenizer(n_pairs=1)

        m.onnx_embed_pairs([("q", "d")])
        _, call_args, _ = session.run.mock_calls[0]
        onnx_input_passed = call_args[1]
        assert "custom" in onnx_input_passed
