import os
from collections.abc import Iterable
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from qwen3_embed.common.onnx_model import (
    EmbeddingWorker,
    OnnxModel,
    OnnxOutputContext,
    OnnxSessionConfig,
)
from qwen3_embed.common.types import Device

# Production code builds the model path via ``Path(model_dir) / model_file``,
# which renders with the platform-native separator (``\`` on Windows, ``/`` on
# POSIX). Using ``os.path.join`` keeps the assertion accurate on every OS.
EXPECTED_MODEL_PATH = os.path.join("dummy", "model.onnx")


# Concrete implementation for testing OnnxModel
class ConcreteOnnxModel(OnnxModel[Any]):
    def _get_worker_class(cls) -> Any:
        return MagicMock()  # type: ignore[return-value]

    def _post_process_onnx_output(self, output: OnnxOutputContext, **kwargs: Any) -> Iterable[Any]:
        return []

    def load_onnx_model(self) -> None:
        pass

    def onnx_embed(self, *args: Any, **kwargs: Any) -> OnnxOutputContext:
        return OnnxOutputContext(model_output=np.zeros((1, 4)))


@pytest.fixture
def model():
    return ConcreteOnnxModel()


@pytest.fixture
def mock_ort():
    with (
        patch("qwen3_embed.common.onnx_model.ort") as mock,
        patch("qwen3_embed.common.onnx_model.load_tokenizer", return_value=(MagicMock(), {})),
    ):
        # Default behavior: CPU provider available, no CUDA
        mock.get_available_providers.return_value = ["CPUExecutionProvider"]
        mock.SessionOptions.return_value = MagicMock()
        mock.GraphOptimizationLevel.ORT_ENABLE_ALL = 99
        mock.ExecutionMode.ORT_PARALLEL = 1

        # Default session mock
        session_mock = MagicMock()
        session_mock.get_providers.return_value = ["CPUExecutionProvider"]
        session_mock.get_inputs.return_value = []
        mock.InferenceSession.return_value = session_mock

        yield mock


def test_load_defaults(model: ConcreteOnnxModel, mock_ort):
    """Test default behavior (CPU fallback)."""
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

    model._load_onnx_model(Path("dummy"), "model.onnx", OnnxSessionConfig(threads=None))

    # Should use CPUExecutionProvider
    mock_ort.InferenceSession.assert_called_with(
        EXPECTED_MODEL_PATH,
        providers=["CPUExecutionProvider"],
        sess_options=mock_ort.SessionOptions.return_value,
    )


def test_load_cuda_explicit(model: ConcreteOnnxModel, mock_ort):
    """Test explicit CUDA request."""
    mock_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    # Mock session to report CUDA provider present
    mock_ort.InferenceSession.return_value.get_providers.return_value = ["CUDAExecutionProvider"]

    model._load_onnx_model(Path("dummy"), "model.onnx", OnnxSessionConfig(threads=None, cuda=True))

    mock_ort.InferenceSession.assert_called_with(
        EXPECTED_MODEL_PATH,
        providers=["CUDAExecutionProvider"],
        sess_options=mock_ort.SessionOptions.return_value,
    )


def test_load_cuda_auto_available(model: ConcreteOnnxModel, mock_ort):
    """Test AUTO CUDA when CUDA is available."""
    mock_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    # Mock session to report CUDA provider present
    mock_ort.InferenceSession.return_value.get_providers.return_value = ["CUDAExecutionProvider"]

    model._load_onnx_model(
        Path("dummy"), "model.onnx", OnnxSessionConfig(threads=None, cuda=Device.AUTO)
    )

    mock_ort.InferenceSession.assert_called_with(
        EXPECTED_MODEL_PATH,
        providers=["CUDAExecutionProvider"],
        sess_options=mock_ort.SessionOptions.return_value,
    )


def test_load_cuda_auto_unavailable(model: ConcreteOnnxModel, mock_ort):
    """Test AUTO CUDA when CUDA is NOT available (fallback to CPU)."""
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

    model._load_onnx_model(
        Path("dummy"), "model.onnx", OnnxSessionConfig(threads=None, cuda=Device.AUTO)
    )

    # Should fallback to CPU
    mock_ort.InferenceSession.assert_called_with(
        EXPECTED_MODEL_PATH,
        providers=["CPUExecutionProvider"],
        sess_options=mock_ort.SessionOptions.return_value,
    )


def test_load_explicit_providers(model: ConcreteOnnxModel, mock_ort):
    """Test explicit providers list."""
    mock_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    # Mock session to report CUDA provider present
    mock_ort.InferenceSession.return_value.get_providers.return_value = ["CUDAExecutionProvider"]

    model._load_onnx_model(
        Path("dummy"),
        "model.onnx",
        OnnxSessionConfig(threads=None, providers=["CUDAExecutionProvider"]),
    )

    mock_ort.InferenceSession.assert_called_with(
        EXPECTED_MODEL_PATH,
        providers=["CUDAExecutionProvider"],
        sess_options=mock_ort.SessionOptions.return_value,
    )


def test_load_providers_validation_error(model: ConcreteOnnxModel, mock_ort):
    """Test validation error for unavailable providers."""
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

    with pytest.raises(ValueError, match="Provider CUDAExecutionProvider is not available"):
        model._load_onnx_model(
            Path("dummy"),
            "model.onnx",
            OnnxSessionConfig(threads=None, providers=["CUDAExecutionProvider"]),
        )


def test_load_cuda_and_providers_warning(model: ConcreteOnnxModel, mock_ort):
    """Test warning when both cuda and providers are specified."""
    mock_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    # Mock session to report CUDA provider present
    mock_ort.InferenceSession.return_value.get_providers.return_value = ["CUDAExecutionProvider"]

    with pytest.warns(UserWarning, match="`cuda` and `providers` are mutually exclusive"):
        model._load_onnx_model(
            Path("dummy"),
            "model.onnx",
            OnnxSessionConfig(
                threads=None,
                cuda=True,
                providers=["CUDAExecutionProvider"],
            ),
        )


def test_load_dml_auto(model: ConcreteOnnxModel, mock_ort):
    """Test DML fallback when CUDA is not available."""
    mock_ort.get_available_providers.return_value = [
        "DmlExecutionProvider",
        "CPUExecutionProvider",
    ]

    model._load_onnx_model(
        Path("dummy"), "model.onnx", OnnxSessionConfig(threads=None, cuda=Device.AUTO)
    )

    mock_ort.InferenceSession.assert_called_with(
        EXPECTED_MODEL_PATH,
        providers=["DmlExecutionProvider"],
        sess_options=mock_ort.SessionOptions.return_value,
    )


def test_load_threads(model: ConcreteOnnxModel, mock_ort):
    """Test thread configuration."""
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

    model._load_onnx_model(Path("dummy"), "model.onnx", OnnxSessionConfig(threads=4))

    so = mock_ort.SessionOptions.return_value
    assert so.intra_op_num_threads == 4
    assert so.inter_op_num_threads == 4


def test_load_parallel_execution(model: ConcreteOnnxModel, mock_ort):
    """Test parallel execution configuration."""
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

    model._load_onnx_model(
        Path("dummy"), "model.onnx", OnnxSessionConfig(threads=None, parallel_execution=True)
    )

    so = mock_ort.SessionOptions.return_value
    assert so.execution_mode == mock_ort.ExecutionMode.ORT_PARALLEL


def test_load_extra_session_options(model: ConcreteOnnxModel, mock_ort):
    """Test extra session options."""
    mock_ort.get_available_providers.return_value = ["CPUExecutionProvider"]

    model._load_onnx_model(
        Path("dummy"),
        "model.onnx",
        OnnxSessionConfig(
            threads=None,
            extra_session_options={"enable_cpu_mem_arena": False},
        ),
    )

    so = mock_ort.SessionOptions.return_value
    assert so.enable_cpu_mem_arena is False


def test_load_cuda_warning(model: ConcreteOnnxModel, mock_ort):
    """Test warning when CUDA is requested but session doesn't use it."""
    mock_ort.get_available_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    # Simulate a scenario where we request CUDA, but the created session
    # ends up NOT using it (e.g. driver issues, although we just mock get_providers here)
    session_mock = MagicMock()
    session_mock.get_providers.return_value = ["CPUExecutionProvider"]  # CUDA missing
    mock_ort.InferenceSession.return_value = session_mock

    with pytest.warns(RuntimeWarning, match="Attempt to set CUDAExecutionProvider failed"):
        model._load_onnx_model(
            Path("dummy"), "model.onnx", OnnxSessionConfig(threads=None, cuda=True)
        )


def test_add_extra_session_options():
    session_options = MagicMock()

    # Test valid options
    ConcreteOnnxModel.add_extra_session_options(session_options, {"enable_cpu_mem_arena": False})
    assert session_options.enable_cpu_mem_arena is False

    # Test valid options (True)
    ConcreteOnnxModel.add_extra_session_options(session_options, {"enable_cpu_mem_arena": True})
    assert session_options.enable_cpu_mem_arena is True

    # Test invalid option
    with pytest.raises(
        ValueError,
        match="invalid_option is unknown or not exposed \\(exposed options: \\('enable_cpu_mem_arena',\\)\\)",
    ):
        ConcreteOnnxModel.add_extra_session_options(session_options, {"invalid_option": True})


def _shape_node(name: str, shape: list[Any]) -> MagicMock:
    """A stand-in for an onnxruntime NodeArg carrying a declared shape."""
    node = MagicMock()
    node.name = name
    node.shape = shape
    return node


class TestDetectStaticBatchSize:
    """A literal on the batch axis pins the graph; a dim_param string does not."""

    def test_fully_dynamic_returns_none(self):
        session = MagicMock()
        session.get_inputs.return_value = [
            _shape_node("input_ids", ["batch_size", "sequence_length"]),
        ]
        session.get_outputs.return_value = [
            _shape_node("last_hidden_state", ["batch_size", "sequence_length", 768]),
        ]
        assert OnnxModel._detect_static_batch_size(session) is None

    def test_literal_on_output_only(self):
        """Shapes as declared by n24q02m/Qwen3-Embedding-0.6B-ONNX model_quantized.onnx:
        the inputs still say ``batch_size`` while the output says ``1``."""
        session = MagicMock()
        session.get_inputs.return_value = [
            _shape_node("input_ids", ["batch_size", "sequence_length"]),
            _shape_node("attention_mask", ["batch_size", "sequence_length"]),
        ]
        session.get_outputs.return_value = [
            _shape_node("last_hidden_state", [1, "sequence_length", 1024]),
        ]
        assert OnnxModel._detect_static_batch_size(session) == 1

    def test_literal_on_input(self):
        session = MagicMock()
        session.get_inputs.return_value = [
            _shape_node("input_ids", [4, "sequence_length"]),
        ]
        session.get_outputs.return_value = [
            _shape_node("last_hidden_state", ["batch_size", "sequence_length", 768]),
        ]
        assert OnnxModel._detect_static_batch_size(session) == 4

    def test_smallest_literal_wins(self):
        session = MagicMock()
        session.get_inputs.return_value = [
            _shape_node("input_ids", [4, "sequence_length"]),
        ]
        session.get_outputs.return_value = [
            _shape_node("last_hidden_state", [1, "sequence_length", 768]),
        ]
        assert OnnxModel._detect_static_batch_size(session) == 1

    def test_rank_one_node_is_not_a_batch_axis(self):
        session = MagicMock()
        session.get_inputs.return_value = [
            _shape_node("input_ids", ["batch_size", "sequence_length"]),
            _shape_node("max_new_tokens", [1]),
        ]
        session.get_outputs.return_value = [
            _shape_node("last_hidden_state", ["batch_size", "sequence_length", 768]),
        ]
        assert OnnxModel._detect_static_batch_size(session) is None


def test_load_records_static_batch_size(model: ConcreteOnnxModel, mock_ort):
    """The batch axis is read once, when the session is created."""
    session_mock = mock_ort.InferenceSession.return_value
    session_mock.get_inputs.return_value = [
        _shape_node("input_ids", ["batch_size", "sequence_length"]),
    ]
    session_mock.get_outputs.return_value = [
        _shape_node("last_hidden_state", [1, "sequence_length", 1024]),
    ]

    model._load_onnx_model(Path("dummy"), "model.onnx", OnnxSessionConfig(threads=None))

    assert model.static_batch_size == 1


def test_preprocess_onnx_input(model: ConcreteOnnxModel):
    """Test _preprocess_onnx_input returns unchanged input by default."""
    data = {"input_ids": np.array([[1, 2, 3]])}
    assert model._preprocess_onnx_input(data) is data


def test_select_exposed_session_options():
    """Test _select_exposed_session_options filters correctly."""
    kwargs = {
        "enable_cpu_mem_arena": True,
        "other_param": "value",
    }
    result = ConcreteOnnxModel._select_exposed_session_options(kwargs)
    assert result == {"enable_cpu_mem_arena": True}


def test_abstract_methods_raise():
    """Test that abstract methods in OnnxModel raise NotImplementedError."""

    class AbstractModel(OnnxModel[Any]):
        pass

    model = AbstractModel()

    with pytest.raises(NotImplementedError):
        AbstractModel._get_worker_class()

    with pytest.raises(NotImplementedError):
        model._post_process_onnx_output(OnnxOutputContext(model_output=np.zeros((1, 4))))

    with pytest.raises(NotImplementedError):
        model.load_onnx_model()

    with pytest.raises(NotImplementedError):
        model.onnx_embed()


# ===========================================================================
# EmbeddingWorker Tests
# ===========================================================================


# Concrete implementation for testing EmbeddingWorker
class ConcreteWorker(EmbeddingWorker[Any]):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> OnnxModel[Any]:
        return ConcreteOnnxModel()

    def process(self, items: Iterable[tuple[int, Any]]) -> Iterable[tuple[int, Any]]:
        yield from items


def test_worker_init():
    """Test EmbeddingWorker initialization."""
    worker = ConcreteWorker(model_name="test_model", cache_dir="/tmp/cache")
    assert isinstance(worker.model, ConcreteOnnxModel)


def test_worker_start():
    """Test EmbeddingWorker.start class method."""
    worker = ConcreteWorker.start(model_name="test_model", cache_dir="/tmp/cache")
    assert isinstance(worker, ConcreteWorker)
    assert isinstance(worker.model, ConcreteOnnxModel)


def test_worker_abstract_methods_raise():
    """Test that abstract methods in EmbeddingWorker raise NotImplementedError."""

    class AbstractWorker(EmbeddingWorker[Any]):
        pass

    # We can't even init it because init_embedding is abstract and called in __init__
    with pytest.raises(NotImplementedError):
        AbstractWorker(model_name="test", cache_dir="/tmp")

    # Mock init_embedding to test process
    with patch.object(AbstractWorker, "init_embedding", return_value=MagicMock()):
        worker = AbstractWorker(model_name="test", cache_dir="/tmp")
        with pytest.raises(NotImplementedError):
            list(worker.process([]))
