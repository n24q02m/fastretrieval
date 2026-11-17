"""Image embedding stack driven from a local model directory under ``tmp_path``.

Each model directory holds a real ONNX graph and a real ``preprocessor_config.json``.
Only the ``onnxruntime`` session is replaced (boundary a); the model download is
bypassed by ``specific_model_path`` (boundary b), so no download fake is needed.
"""

import io
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Pillow ships in the optional [image] extra. The integration job installs only the dev
# group and still collects every module, so this module skips there instead of failing.
pytest.importorskip("PIL.Image", reason="Pillow belongs to the optional [image] extra")

from PIL import Image

import fastretrieval.common.onnx_model as onnx_model_module
from fastretrieval.common.onnx_model import OnnxOutputContext
from fastretrieval.common.types import Device, NumpyArray
from fastretrieval.contract import PreprocessorSpec
from fastretrieval.models.image.image_embedding import ImageEmbedding
from fastretrieval.models.image.image_embedding_base import ImageEmbeddingBase
from fastretrieval.models.image.onnx_embedding import (
    OnnxImageEmbedding,
    OnnxImageEmbeddingWorker,
    supported_onnx_models,
)
from fastretrieval.models.image.onnx_image_model import OnnxImageModel

MODEL_NAME = "Qdrant/clip-ViT-B-32-vision"
SIDE = 4
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
WHITE = (255, 255, 255)
CLIP_CONFIG = {
    "image_processor_type": "CLIPImageProcessor",
    "size": {"shortest_edge": SIDE},
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}


# --- onnxruntime stand-in (boundary a) ---------------------------------------


class _StubNode:
    def __init__(self, name: str, shape: list[Any]) -> None:
        self.name = name
        self.shape = shape


class _StubSessionOptions:
    """Attribute bag in place of ``onnxruntime.SessionOptions``."""


class _StubInferenceSession:
    """Deterministic ``onnxruntime.InferenceSession``.

    Pixel feeds return each image's per-channel mean; raw byte feeds return
    ``[len(bytes), 1.0]``. Both depend only on the input, so outputs are checkable.
    """

    input_names: tuple[str, ...] = ("pixel_values",)

    def __init__(self, path: str, providers: Any = None, sess_options: Any = None) -> None:
        self.path = path
        self.providers = providers
        self.sess_options = sess_options
        self.feeds: list[dict[str, Any]] = []
        _StubOrt.sessions.append(self)

    def get_inputs(self) -> list[_StubNode]:
        return [_StubNode(name, ["batch", 3, SIDE, SIDE]) for name in self.input_names]

    def get_outputs(self) -> list[_StubNode]:
        return [_StubNode("image_embeds", ["batch", 3])]

    def get_providers(self) -> list[str]:
        return [p if isinstance(p, str) else p[0] for p in self.providers]

    def _compute(self, batch: np.ndarray) -> np.ndarray:
        if batch.dtype == object:
            return np.asarray([[len(item), 1.0] for item in batch], dtype=np.float32)
        return batch.mean(axis=(2, 3))

    def run(self, output_names: Any, feeds: dict[str, Any]) -> list[np.ndarray]:
        self.feeds.append(feeds)
        (batch,) = feeds.values()
        return [self._compute(batch)]


class _HalfPrecisionSession(_StubInferenceSession):
    """Returns float16 output with an extra singleton axis, like some exported encoders."""

    def run(self, output_names: Any, feeds: dict[str, Any]) -> list[np.ndarray]:
        (batch,) = feeds.values()
        return [self._compute(batch)[:, None, :].astype(np.float16)]


class _InputlessSession(_StubInferenceSession):
    input_names = ()


class _StubOrt:
    """Stand-in for the ``onnxruntime`` module namespace."""

    class GraphOptimizationLevel:
        ORT_ENABLE_ALL = "ORT_ENABLE_ALL"

    class ExecutionMode:
        ORT_PARALLEL = "ORT_PARALLEL"

    SessionOptions = _StubSessionOptions
    InferenceSession: type[_StubInferenceSession] = _StubInferenceSession
    available_providers: tuple[str, ...] = ("CPUExecutionProvider",)
    sessions: list[_StubInferenceSession] = []

    @classmethod
    def get_available_providers(cls) -> list[str]:
        return list(cls.available_providers)


@pytest.fixture
def stub_ort(monkeypatch: pytest.MonkeyPatch) -> type[_StubOrt]:
    # Boundary (a): onnxruntime.InferenceSession is stubbed; CPython 3.15 has no wheel.
    monkeypatch.setattr(onnx_model_module, "ort", _StubOrt)
    monkeypatch.setattr(_StubOrt, "sessions", [])
    return _StubOrt


# --- pool plumbing for the multiprocessing path ------------------------------


class _StubOrtWorker(OnnxImageEmbeddingWorker):
    """Real image worker that installs the session stub inside its own process."""

    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> OnnxImageEmbedding:
        # Boundary (a): a spawned interpreter never sees the parent's monkeypatch and
        # CPython 3.15 has no onnxruntime wheel, so the worker installs the stub itself.
        pytest.MonkeyPatch().setattr(onnx_model_module, "ort", _StubOrt)
        return super().init_embedding(model_name, cache_dir, **kwargs)


class _StubOrtPoolEmbedding(OnnxImageEmbedding):
    """Real embedding whose pool starts ``_StubOrtWorker`` processes."""

    @classmethod
    def _get_worker_class(cls) -> type[OnnxImageEmbeddingWorker]:
        return _StubOrtWorker


# --- local model directory ---------------------------------------------------


def _write_graph(path: Path, extension_domain: str | None = None) -> None:
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["pixel_values"], ["image_embeds"])],
        "probe",
        [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, ["batch", 3])],
        [helper.make_tensor_value_info("image_embeds", TensorProto.FLOAT, ["batch", 3])],
    )
    opsets = [helper.make_opsetid("", 17)]
    if extension_domain is not None:
        opsets.append(helper.make_opsetid(extension_domain, 1))
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(helper.make_model(graph, opset_imports=opsets), str(path))


def _make_model_dir(
    root: Path,
    model_file: str = "model.onnx",
    config: dict[str, Any] | None = CLIP_CONFIG,
    extension_domain: str | None = None,
) -> Path:
    model_dir = root / "model"
    _write_graph(model_dir / model_file, extension_domain)
    if config is not None:
        config_path = model_dir / "preprocessor_config.json"
        config_path.write_text(json.dumps(config), encoding="utf-8")
    return model_dir


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    return _make_model_dir(tmp_path)


def _embedding(
    model_dir: Path, tmp_path: Path, model_name: str = MODEL_NAME, **kwargs: Any
) -> OnnxImageEmbedding:
    # Boundary (b): specific_model_path makes download_model return the local directory.
    return OnnxImageEmbedding(
        model_name,
        cache_dir=str(tmp_path / "cache"),
        specific_model_path=str(model_dir),
        **kwargs,
    )


def _solid(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (SIDE, SIDE), color)


def _expected(color: tuple[int, int, int]) -> np.ndarray:
    # mean = std = 0.5 maps a 255 channel to +1 and a 0 channel to -1; L2 norm is sqrt(3).
    return np.array([1.0 if value == 255 else -1.0 for value in color]) / np.sqrt(3)


def _unit(vector: list[float]) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float64)
    return array / np.linalg.norm(array)


def _session(model: OnnxImageModel[Any]) -> _StubInferenceSession:
    """The loaded session, which the stub_ort fixture made a _StubInferenceSession."""
    session = model.model
    assert isinstance(session, _StubInferenceSession)
    return session


class _ProbeImageModel(OnnxImageModel[np.ndarray]):
    """Smallest concrete runner: loads the local graph, returns raw rows."""

    declared_preprocessor: PreprocessorSpec | None

    def __init__(self, model_dir: Path, model_file: str = "model.onnx") -> None:
        super().__init__()
        self.model_dir = model_dir
        self.model_file = model_file
        self.load_calls = 0

    def load_onnx_model(self) -> None:
        self.load_calls += 1
        self._load_onnx_model(self.model_dir, self.model_file)

    def _post_process_onnx_output(self, output: OnnxOutputContext, **kwargs: Any) -> list[Any]:
        return list(output.model_output)


# --- OnnxImageModel ------------------------------------------------------------


def test_image_model_base_leaves_worker_postprocess_and_loader_to_subclasses():
    base = OnnxImageModel()

    with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
        OnnxImageModel._get_worker_class()
    with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
        base._post_process_onnx_output(OnnxOutputContext(model_output=np.zeros(1)))
    with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
        base.load_onnx_model()


def test_image_model_starts_without_session_or_preprocessor():
    base = OnnxImageModel()

    assert base.model is None
    assert base.preprocessor_spec is None


def test_preprocess_onnx_input_returns_the_feed_unchanged():
    feeds: dict[str, NumpyArray] = {"pixel_values": np.zeros((1, 3, 1, 1), dtype=np.float32)}

    assert OnnxImageModel()._preprocess_onnx_input(feeds, extra=1) is feeds


def test_load_reads_the_image_spec_from_the_config_file(stub_ort, model_dir):
    runner = _ProbeImageModel(model_dir)

    session, input_names = runner._load_onnx_model(model_dir, "model.onnx")

    assert session is runner.model
    assert Path(session.path) == model_dir / "model.onnx"
    assert input_names == ["pixel_values"]
    assert runner.model_input_names == {"pixel_values"}
    assert runner.static_batch_size is None
    assert runner.preprocessor_spec == PreprocessorSpec(
        kind="image",
        config_file="preprocessor_config.json",
        image_size=(SIDE, SIDE),
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )


def test_load_prefers_a_declared_spec_over_the_config_file(stub_ort, model_dir):
    runner = _ProbeImageModel(model_dir)
    runner.declared_preprocessor = PreprocessorSpec(kind="image", image_size=(2, 2))

    runner.load_onnx_model()

    assert runner.preprocessor_spec == PreprocessorSpec(kind="image", image_size=(2, 2))


def test_load_falls_back_to_image_defaults_without_config_or_declaration(stub_ort, tmp_path):
    runner = _ProbeImageModel(_make_model_dir(tmp_path, config=None))

    runner.load_onnx_model()

    assert runner.preprocessor_spec == PreprocessorSpec(kind="image")


def test_text_only_config_file_is_kept_and_rejected_when_encoding(stub_ort, tmp_path):
    runner = _ProbeImageModel(_make_model_dir(tmp_path, config={"do_lower_case": True}))
    runner.load_onnx_model()

    assert runner.preprocessor_spec == PreprocessorSpec(
        kind="text", config_file="preprocessor_config.json"
    )
    with pytest.raises(ValueError, match="kind='image' or kind='none', got 'text'"):
        runner.onnx_embed([_solid(RED)])


def test_declared_text_spec_is_kept_even_without_a_config_file(stub_ort, tmp_path):
    runner = _ProbeImageModel(_make_model_dir(tmp_path, config=None))
    runner.declared_preprocessor = PreprocessorSpec(kind="text")

    runner.load_onnx_model()

    assert runner.preprocessor_spec == PreprocessorSpec(kind="text")


@pytest.mark.parametrize("domain", ["ai.onnx.contrib", "com.microsoft.extensions"])
def test_self_contained_graph_switches_to_raw_bytes(stub_ort, tmp_path, domain):
    runner = _ProbeImageModel(_make_model_dir(tmp_path, extension_domain=domain))
    runner.declared_preprocessor = PreprocessorSpec(kind="image")

    runner.load_onnx_model()

    assert runner.preprocessor_spec == PreprocessorSpec(kind="none")


def test_raw_image_bytes_copies_byte_like_inputs():
    assert OnnxImageModel._raw_image_bytes(b"abc") == b"abc"
    assert OnnxImageModel._raw_image_bytes(bytearray(b"abc")) == b"abc"
    assert OnnxImageModel._raw_image_bytes(memoryview(b"abc")) == b"abc"
    assert type(OnnxImageModel._raw_image_bytes(bytearray(b"x"))) is bytes


def test_raw_image_bytes_reads_str_and_path_files(tmp_path):
    path = tmp_path / "image.bin"
    path.write_bytes(b"\x00\x01payload")

    assert OnnxImageModel._raw_image_bytes(str(path)) == b"\x00\x01payload"
    assert OnnxImageModel._raw_image_bytes(path) == b"\x00\x01payload"


def test_raw_image_bytes_encodes_an_image_object_as_png():
    pixels = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

    payload = OnnxImageModel._raw_image_bytes(Image.fromarray(pixels))

    assert payload.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(payload)) as decoded:
        np.testing.assert_array_equal(np.asarray(decoded), pixels)


def test_raw_image_bytes_rejects_unsupported_inputs():
    with pytest.raises(TypeError, match="unsupported raw image input: int"):
        OnnxImageModel._raw_image_bytes(42)


def test_is_image_object_detects_convert_or_save():
    class _Saveable:
        def save(self, *args: Any) -> None:
            return None

    assert OnnxImageModel._is_image_object(_solid(RED))
    assert OnnxImageModel._is_image_object(_Saveable())
    assert not OnnxImageModel._is_image_object(b"bytes")


def test_encode_requires_a_resolved_preprocessor(model_dir):
    with pytest.raises(ValueError, match="Image preprocessor is not initialized"):
        _ProbeImageModel(model_dir)._encode_images([_solid(RED)])


def test_unloaded_model_cannot_build_inputs_or_embed(model_dir):
    runner = _ProbeImageModel(model_dir)

    with pytest.raises(ValueError, match=r"Model not loaded\. Please call load_onnx_model"):
        runner._build_onnx_input(np.zeros((1, 3, SIDE, SIDE), dtype=np.float32))
    with pytest.raises(ValueError, match=r"Model not loaded\. Please call load_onnx_model"):
        runner.onnx_embed([_solid(RED)])


def test_graph_without_inputs_is_rejected(stub_ort, monkeypatch, model_dir):
    monkeypatch.setattr(_StubOrt, "InferenceSession", _InputlessSession)
    runner = _ProbeImageModel(model_dir)
    runner.load_onnx_model()

    with pytest.raises(ValueError, match="Image model has no ONNX inputs"):
        runner.onnx_embed([_solid(RED)])


def test_onnx_embed_feeds_normalized_pixels_to_the_first_input(stub_ort, model_dir):
    runner = _ProbeImageModel(model_dir)
    runner.load_onnx_model()

    context = runner.onnx_embed([_solid(RED), _solid(WHITE)])

    (feed,) = _session(runner).feeds
    assert list(feed) == ["pixel_values"]
    assert feed["pixel_values"].shape == (2, 3, SIDE, SIDE)
    assert feed["pixel_values"].dtype == np.float32
    np.testing.assert_allclose(context.model_output, [[1, -1, -1], [1, 1, 1]], atol=1e-6)


def test_onnx_embed_casts_float16_output_and_flattens_per_image(stub_ort, monkeypatch, model_dir):
    monkeypatch.setattr(_StubOrt, "InferenceSession", _HalfPrecisionSession)
    runner = _ProbeImageModel(model_dir)
    runner.load_onnx_model()

    context = runner.onnx_embed([_solid(BLUE), _solid(GREEN)])

    assert context.model_output.dtype == np.float32
    assert context.model_output.shape == (2, 3)
    np.testing.assert_allclose(context.model_output, [[-1, -1, 1], [-1, 1, -1]], atol=1e-3)


def test_embed_images_loads_the_model_once_on_first_use(stub_ort, model_dir):
    runner = _ProbeImageModel(model_dir)

    first = list(runner._embed_images(MODEL_NAME, "unused", [_solid(RED)]))
    second = list(runner._embed_images(MODEL_NAME, "unused", _solid(BLUE)))

    assert runner.load_calls == 1
    assert len(stub_ort.sessions) == 1
    np.testing.assert_allclose(first, [[1, -1, -1]], atol=1e-6)
    np.testing.assert_allclose(second, [[-1, -1, 1]], atol=1e-6)


# --- OnnxImageEmbedding ------------------------------------------------------------


def test_embedding_loads_eagerly_with_threads_providers_and_session_options(
    stub_ort, model_dir, tmp_path
):
    emb = _embedding(
        model_dir,
        tmp_path,
        threads=2,
        providers=["CPUExecutionProvider"],
        enable_cpu_mem_arena=False,
    )

    (session,) = stub_ort.sessions
    assert emb.model is session
    assert Path(session.path) == model_dir / "model.onnx"
    assert session.providers == ["CPUExecutionProvider"]
    assert session.sess_options.intra_op_num_threads == 2
    assert session.sess_options.inter_op_num_threads == 2
    assert session.sess_options.enable_cpu_mem_arena is False
    assert emb._model_dir == model_dir
    assert emb.cache_dir == str(tmp_path / "cache")
    assert (tmp_path / "cache").is_dir()
    assert emb.preprocessor_spec is not None
    assert emb.preprocessor_spec.image_size == (SIDE, SIDE)


def test_embedding_ignores_unexposed_keyword_arguments(stub_ort, model_dir, tmp_path):
    emb = _embedding(model_dir, tmp_path, unrelated_option=True)

    assert emb._extra_session_options == {}
    assert not hasattr(stub_ort.sessions[0].sess_options, "enable_cpu_mem_arena")


def test_lazy_embedding_opens_the_session_only_when_asked(stub_ort, model_dir, tmp_path):
    emb = _embedding(model_dir, tmp_path, lazy_load=True)

    assert stub_ort.sessions == []
    emb.load_onnx_model()

    assert emb.model is stub_ort.sessions[0]
    np.testing.assert_allclose(list(emb.embed(_solid(RED))), [_expected(RED)], atol=1e-6)


def test_first_device_id_selects_the_cuda_device(stub_ort, monkeypatch, model_dir, tmp_path):
    monkeypatch.setattr(
        _StubOrt, "available_providers", ("CUDAExecutionProvider", "CPUExecutionProvider")
    )

    emb = _embedding(model_dir, tmp_path, device_ids=[3, 5])

    assert emb.device_id == 3
    assert _session(emb).providers == [("CUDAExecutionProvider", {"device_id": 3})]


def test_explicit_device_id_wins_over_device_ids(stub_ort, model_dir, tmp_path):
    emb = _embedding(model_dir, tmp_path, device_id=7, device_ids=[3], lazy_load=True)

    assert emb.device_id == 7
    assert emb.device_ids == [3]


def test_embedding_without_device_ids_keeps_device_id_unset(stub_ort, model_dir, tmp_path):
    assert _embedding(model_dir, tmp_path, lazy_load=True).device_id is None


def test_jina_vision_model_loads_its_nested_model_file(stub_ort, tmp_path):
    model_dir = _make_model_dir(tmp_path, model_file="onnx/vision_model.onnx")

    emb = _embedding(model_dir, tmp_path, model_name="jinaai/jina-clip-v1")

    assert Path(_session(emb).path) == model_dir / "onnx" / "vision_model.onnx"
    np.testing.assert_allclose(list(emb.embed(_solid(GREEN))), [_expected(GREEN)], atol=1e-6)


def test_embed_batches_images_and_returns_unit_vectors_in_order(stub_ort, model_dir, tmp_path):
    emb = _embedding(model_dir, tmp_path)
    colors = [RED, GREEN, BLUE]

    vectors = list(emb.embed([_solid(color) for color in colors], batch_size=2))

    assert [feed["pixel_values"].shape[0] for feed in _session(emb).feeds] == [2, 1]
    np.testing.assert_allclose(vectors, [_expected(color) for color in colors], atol=1e-6)


def test_embed_accepts_a_generator_of_images(stub_ort, model_dir, tmp_path):
    emb = _embedding(model_dir, tmp_path)

    vectors = list(emb.embed((_solid(color) for color in (WHITE, BLUE)), batch_size=1))

    np.testing.assert_allclose(vectors, [_expected(WHITE), _expected(BLUE)], atol=1e-6)


def test_embed_wraps_a_single_str_or_path_image_file(stub_ort, model_dir, tmp_path):
    path = tmp_path / "blue.png"
    _solid(BLUE).save(path)
    emb = _embedding(model_dir, tmp_path)

    from_str = list(emb.embed(str(path)))
    from_path = list(emb.embed(path))

    np.testing.assert_allclose(from_str, [_expected(BLUE)], atol=1e-6)
    np.testing.assert_allclose(from_path, [_expected(BLUE)], atol=1e-6)


def test_embed_honours_a_declared_preprocessor(stub_ort, model_dir, tmp_path):
    declared = PreprocessorSpec(
        kind="image", image_size=(2, 3), image_mean=(0.0, 0.0, 0.0), image_std=(1.0, 1.0, 1.0)
    )
    emb = _embedding(model_dir, tmp_path, preprocessor=declared)

    vectors = list(emb.embed(_solid(RED)))

    assert emb.preprocessor_spec is declared
    assert _session(emb).feeds[0]["pixel_values"].shape == (1, 3, 2, 3)
    np.testing.assert_allclose(vectors, [[1.0, 0.0, 0.0]], atol=1e-6)


@pytest.mark.parametrize(
    ("raw", "length"),
    [(b"abc", 3), (bytearray(b"abcd"), 4), (memoryview(b"ab"), 2)],
)
def test_self_contained_graph_embeds_single_raw_byte_inputs(stub_ort, tmp_path, raw, length):
    model_dir = _make_model_dir(tmp_path, extension_domain="ai.onnx.contrib")
    emb = _embedding(model_dir, tmp_path)

    vectors = list(emb.embed(raw))

    (feed,) = _session(emb).feeds
    assert feed["pixel_values"].dtype == object
    assert feed["pixel_values"].tolist() == [bytes(raw)]
    np.testing.assert_allclose(vectors, [_unit([length, 1.0])], atol=1e-6)


def test_self_contained_graph_receives_png_bytes_for_an_image_object(stub_ort, tmp_path):
    model_dir = _make_model_dir(tmp_path, extension_domain="com.microsoft.extensions")
    emb = _embedding(model_dir, tmp_path)
    image = _solid(WHITE)

    vectors = list(emb.embed(image))

    png = OnnxImageModel._raw_image_bytes(image)
    assert _session(emb).feeds[0]["pixel_values"].tolist() == [png]
    np.testing.assert_allclose(vectors, [_unit([len(png), 1.0])], atol=1e-6)


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        (np.array([3.0, 4.0]), [[0.6, 0.8]]),
        (np.array([[3.0, 4.0], [0.0, 2.0]]), [[0.6, 0.8], [0.0, 1.0]]),
        (np.array([[[3.0], [4.0]], [[0.0], [5.0]]]), [[0.6, 0.8], [0.0, 1.0]]),
    ],
)
def test_post_processing_flattens_and_l2_normalizes(
    stub_ort, model_dir, tmp_path, output, expected
):
    emb = _embedding(model_dir, tmp_path, lazy_load=True)

    result = emb._post_process_onnx_output(OnnxOutputContext(model_output=output))

    np.testing.assert_allclose(np.asarray(result), expected)


def test_embedding_uses_the_image_worker_and_model_registry():
    assert OnnxImageEmbedding._get_worker_class() is OnnxImageEmbeddingWorker
    assert OnnxImageEmbedding._list_supported_models() is supported_onnx_models


# --- workers and the multiprocessing path --------------------------------------


def test_worker_builds_a_single_thread_embedding_and_embeds_batches(stub_ort, model_dir, tmp_path):
    worker = OnnxImageEmbeddingWorker.start(
        MODEL_NAME,
        str(tmp_path / "cache"),
        specific_model_path=str(model_dir),
        enable_cpu_mem_arena=False,
    )

    out = list(worker.process(iter([(5, [_solid(RED)]), (2, [_solid(BLUE), _solid(WHITE)])])))

    assert isinstance(worker.model, OnnxImageEmbedding)
    assert worker.model.threads == 1
    assert _session(worker.model).sess_options.intra_op_num_threads == 1
    assert _session(worker.model).sess_options.enable_cpu_mem_arena is False
    assert [idx for idx, _ in out] == [5, 2]
    np.testing.assert_allclose(out[0][1].model_output, [[1, -1, -1]], atol=1e-6)
    np.testing.assert_allclose(out[1][1].model_output, [[-1, -1, 1], [1, 1, 1]], atol=1e-6)


def test_parallel_embed_runs_in_a_worker_process_and_keeps_order(stub_ort, model_dir, tmp_path):
    emb = _StubOrtPoolEmbedding(
        MODEL_NAME,
        cache_dir=str(tmp_path / "cache"),
        specific_model_path=str(model_dir),
        lazy_load=True,
        enable_cpu_mem_arena=False,
    )
    colors = [RED, GREEN, BLUE, WHITE]

    vectors = list(emb.embed([_solid(color) for color in colors], batch_size=1, parallel=1))

    assert stub_ort.sessions == []
    np.testing.assert_allclose(vectors, [_expected(color) for color in colors], atol=1e-6)


def test_parallel_zero_starts_one_worker_per_core(stub_ort, model_dir, tmp_path):
    emb = _StubOrtPoolEmbedding(
        MODEL_NAME,
        cache_dir=str(tmp_path / "cache"),
        specific_model_path=str(model_dir),
        lazy_load=True,
    )
    colors = [WHITE, RED, GREEN]
    assert emb.cache_dir is not None

    vectors = list(
        emb._embed_images(
            model_name=MODEL_NAME,
            cache_dir=emb.cache_dir,
            images=(_solid(color) for color in colors),
            batch_size=2,
            parallel=0,
            specific_model_path=str(model_dir),
        )
    )

    assert stub_ort.sessions == []
    np.testing.assert_allclose(vectors, [_expected(color) for color in colors], atol=1e-6)


# --- ImageEmbedding facade -------------------------------------------------------


def test_facade_lists_every_onnx_model_as_dicts():
    listed = ImageEmbedding.list_supported_models()

    assert listed == [asdict(model) for model in supported_onnx_models]
    assert [entry["model"] for entry in listed] == [
        "Qdrant/clip-ViT-B-32-vision",
        "Qdrant/resnet50-onnx",
        "Qdrant/Unicom-ViT-B-16",
        "Qdrant/Unicom-ViT-B-32",
        "jinaai/jina-clip-v1",
    ]


def test_facade_matches_the_model_name_case_insensitively_and_forwards_options(
    stub_ort, model_dir, tmp_path
):
    facade = ImageEmbedding(
        "QDRANT/clip-vit-b-32-VISION",
        str(tmp_path / "cache"),
        threads=2,
        providers=["CPUExecutionProvider"],
        cuda=Device.CPU,
        device_ids=[1],
        lazy_load=True,
        local_files_only=True,
        specific_model_path=str(model_dir),
    )

    inner = facade.model
    assert isinstance(inner, OnnxImageEmbedding)
    assert inner.model_description.model == MODEL_NAME
    assert (inner.threads, inner.providers, inner.cuda) == (
        2,
        ["CPUExecutionProvider"],
        Device.CPU,
    )
    assert (inner.device_ids, inner.device_id, inner.lazy_load) == ([1], 1, True)
    assert inner._local_files_only is True
    assert facade._local_files_only is True
    assert stub_ort.sessions == []


def test_facade_rejects_an_unknown_model(tmp_path):
    with pytest.raises(ValueError, match="Model acme/vision is not supported in ImageEmbedding"):
        ImageEmbedding("acme/vision", str(tmp_path / "cache"))


def test_facade_embedding_size_is_resolved_once_and_cached(stub_ort, model_dir, tmp_path):
    facade = ImageEmbedding(
        MODEL_NAME, str(tmp_path / "cache"), lazy_load=True, specific_model_path=str(model_dir)
    )

    assert facade._embedding_size is None
    assert facade.embedding_size == 512
    assert facade._embedding_size == 512
    assert facade.embedding_size == 512


@pytest.mark.parametrize(
    ("name", "dim"),
    [
        ("JINAAI/jina-clip-v1", 768),
        ("Qdrant/resnet50-onnx", 2048),
        ("qdrant/unicom-vit-b-16", 768),
    ],
)
def test_facade_reports_the_declared_embedding_size(name, dim):
    assert ImageEmbedding.get_embedding_size(name) == dim


def test_facade_rejects_the_size_of_an_unknown_model():
    with pytest.raises(ValueError, match=r"Embedding size for model acme/vision is unknown\."):
        ImageEmbedding.get_embedding_size("acme/vision")


def test_facade_embed_delegates_to_the_onnx_model(stub_ort, model_dir, tmp_path):
    facade = ImageEmbedding(
        MODEL_NAME, str(tmp_path / "cache"), specific_model_path=str(model_dir)
    )

    vectors = list(facade.embed([_solid(GREEN), _solid(RED)], batch_size=1))

    np.testing.assert_allclose(vectors, [_expected(GREEN), _expected(RED)], atol=1e-6)
    assert isinstance(facade.model, OnnxImageEmbedding)
    assert len(_session(facade.model).feeds) == 2


# --- ImageEmbeddingBase ------------------------------------------------------------


def test_base_stores_settings_and_pops_local_files_only():
    base = ImageEmbeddingBase("m", "cache", 3, local_files_only=True, other=1)

    assert (base.model_name, base.cache_dir, base.threads) == ("m", "cache", 3)
    assert base._local_files_only is True
    assert base._embedding_size is None


def test_base_defaults_local_files_only_to_false():
    assert ImageEmbeddingBase("m")._local_files_only is False


def test_base_leaves_embedding_and_size_to_subclasses():
    base = ImageEmbeddingBase("m")

    with pytest.raises(NotImplementedError):
        base.embed([])
    with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
        ImageEmbeddingBase.get_embedding_size("m")
    with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
        _ = base.embedding_size
