"""Behavior of the ColPali multimodal late-interaction stack.

Covers ColPali, its text and image workers, the LateInteractionMultimodalEmbedding facade, its
base class and the shared OnnxMultimodalModel runner. Every model loads from a local model
directory under tmp_path that holds a real tokenizer, real config files and, where needed, a
real preprocessor_config.json; images are real PIL images. Only the ONNX session is stubbed.
"""

import io
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

# Pillow ships in the optional [image] extra. The integration job installs only the dev
# group and still collects every module, so this module skips there instead of failing.
pytest.importorskip("PIL.Image", reason="Pillow belongs to the optional [image] extra")

from PIL import Image
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

import fastretrieval.common.onnx_model as onnx_model_module
from fastretrieval.common.onnx_model import OnnxOutputContext
from fastretrieval.common.preprocessor_utils import load_tokenizer
from fastretrieval.common.types import NumpyArray
from fastretrieval.contract.preprocessor import PreprocessorSpec
from fastretrieval.models.late_interaction_multimodal import LateInteractionMultimodalEmbedding
from fastretrieval.models.late_interaction_multimodal.colpali import (
    ColPali,
    ColPaliImageEmbeddingWorker,
    ColPaliTextEmbeddingWorker,
    supported_colpali_models,
)
from fastretrieval.models.late_interaction_multimodal.late_interaction_multimodal_embedding_base import (
    LateInteractionMultimodalEmbeddingBase,
)
from fastretrieval.models.late_interaction_multimodal.onnx_multimodal_model import (
    OnnxMultimodalModel,
)

COLPALI = "Qdrant/colpali-v1.3-fp16"
DIM = 128
PATCHES = 2
IMAGE_SIZE = 16
VOCAB = {"<pad>": 0, "<eos>": 1, "<s>": 2, "<unk>": 3, "Query": 4, ":": 5, "hello": 6, "world": 7}
PAD, BOS, COLON, HELLO, WORLD = (VOCAB[token] for token in ("<pad>", "<s>", ":", "hello", "world"))
MARKER = ColPali.QUERY_MARKER_TOKEN_ID
# Every query becomes "<s>Query: <query><pad>x10\n": "<s>", "Query", ":" and ten "<pad>".
EXTENSION = 13
HELLO_IDS = [*MARKER, COLON, HELLO] + [PAD] * 10
TEXT_IMAGE_INPUTS = ("input_ids", "attention_mask", "pixel_values")
WHITE, BLACK, RED = (255, 255, 255), (0, 0, 0), (255, 0, 0)
IMAGE_CONFIG = {
    "image_processor_type": "SiglipImageProcessor",
    "size": {"height": IMAGE_SIZE, "width": IMAGE_SIZE},
    "image_mean": [0.5, 0.5, 0.5],
    "image_std": [0.5, 0.5, 0.5],
}
# With mean 0.5 and std 0.5 a pixel value v maps to v / 127.5 - 1.
NORMALIZED = {WHITE: 1.0, BLACK: -1.0, RED: -1.0 / 3.0}


class _StubSession:
    """Deterministic stand-in for the ColPali ONNX graph.

    An image run (``input_ids`` holding the empty-text placeholder) returns a flat
    ``(batch, PATCHES * DIM)`` row per image filled with that image's mean pixel value, or with
    the byte length of a raw image. A text run returns ``(batch, sequence, DIM)`` whose first
    column is the input id, so tests can read back which token produced every row.
    """

    def __init__(self, input_names: tuple[str, ...], batch_dim: Any, output_dtype: Any) -> None:
        self.input_names = input_names
        self.batch_dim = batch_dim
        self.output_dtype = output_dtype
        self.calls: list[dict[str, np.ndarray]] = []

    def get_inputs(self) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(
                name=name,
                shape=[self.batch_dim, *ColPali.IMAGE_PLACEHOLDER_SIZE]
                if name == "pixel_values"
                else [self.batch_dim, "sequence"],
            )
            for name in self.input_names
        ]

    def get_outputs(self) -> list[SimpleNamespace]:
        return [SimpleNamespace(name="last_hidden_state", shape=[self.batch_dim, "seq", DIM])]

    def get_providers(self) -> list[str]:
        return ["CPUExecutionProvider"]

    def run(self, output_names: Any, inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        self.calls.append({name: np.array(value, copy=True) for name, value in inputs.items()})
        input_ids = inputs["input_ids"]
        if np.array_equal(input_ids[0], ColPali.EMPTY_TEXT_PLACEHOLDER):
            pixels = inputs["pixel_values"]
            if pixels.dtype == object:
                per_image = np.asarray([len(item) for item in pixels], dtype=np.float32)
            else:
                per_image = pixels.reshape(len(pixels), -1).mean(axis=1)
            output = np.repeat(per_image[:, None], PATCHES * DIM, axis=1)
        else:
            output = np.zeros((*input_ids.shape, DIM), dtype=np.float32)
            output[..., 0] = input_ids
        return [output.astype(self.output_dtype)]


class _StubOrt:
    """Module-shaped stand-in for onnxruntime that hands out _StubSession objects."""

    GraphOptimizationLevel = SimpleNamespace(ORT_ENABLE_ALL="ORT_ENABLE_ALL")
    ExecutionMode = SimpleNamespace(ORT_PARALLEL="ORT_PARALLEL")
    SessionOptions = SimpleNamespace

    def __init__(
        self,
        input_names: tuple[str, ...] = TEXT_IMAGE_INPUTS,
        batch_dim: Any = "batch",
        output_dtype: Any = np.float32,
    ) -> None:
        self.input_names = input_names
        self.batch_dim = batch_dim
        self.output_dtype = output_dtype
        self.created: list[dict[str, Any]] = []

    @staticmethod
    def get_available_providers() -> list[str]:
        return ["CPUExecutionProvider"]

    def InferenceSession(self, path: str, providers: Any, sess_options: Any) -> _StubSession:
        self.created.append({"path": path, "providers": providers, "sess_options": sess_options})
        return _StubSession(self.input_names, self.batch_dim, self.output_dtype)


class _StubbedTextWorker(ColPaliTextEmbeddingWorker):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> ColPali:
        # Boundary (a) onnxruntime: a spawned worker starts unpatched, so it installs the stub.
        pytest.MonkeyPatch().setattr(onnx_model_module, "ort", _StubOrt())
        return super().init_embedding(model_name, cache_dir, **kwargs)


class _StubbedImageWorker(ColPaliImageEmbeddingWorker):
    def init_embedding(self, model_name: str, cache_dir: str, **kwargs: Any) -> ColPali:
        # Boundary (a) onnxruntime: a spawned worker starts unpatched, so it installs the stub.
        pytest.MonkeyPatch().setattr(onnx_model_module, "ort", _StubOrt())
        return super().init_embedding(model_name, cache_dir, **kwargs)


class _SpawnableColPali(ColPali):
    """Real ColPali whose worker pool starts the stub-installing workers above."""

    @classmethod
    def _get_text_worker_class(cls) -> type[ColPaliTextEmbeddingWorker]:
        return _StubbedTextWorker

    @classmethod
    def _get_image_worker_class(cls) -> type[ColPaliImageEmbeddingWorker]:
        return _StubbedImageWorker


@pytest.fixture
def install_ort(monkeypatch: pytest.MonkeyPatch) -> Any:
    def install(**kwargs: Any) -> _StubOrt:
        stub = _StubOrt(**kwargs)
        # Boundary (a) onnxruntime.InferenceSession: no cp315 wheel and no real graph in unit tests.
        monkeypatch.setattr(onnx_model_module, "ort", stub)
        return stub

    return install


@pytest.fixture
def stub_ort(install_ort: Any) -> _StubOrt:
    return install_ort()


def _write_model_dir(model_dir: Path, preprocessor_config: dict[str, Any] | None) -> Path:
    model_dir.mkdir(parents=True)
    tokenizer = Tokenizer(WordLevel(VOCAB, unk_token="<unk>"))
    tokenizer.pre_tokenizer = Whitespace()
    tokenizer.save(str(model_dir / "tokenizer.json"))
    files: dict[str, Any] = {
        "config.json": {"pad_token_id": PAD},
        "tokenizer_config.json": {"model_max_length": 64, "pad_token": "<pad>"},
        "special_tokens_map.json": {
            "bos_token": "<s>",
            "pad_token": "<pad>",
            "unk_token": "<unk>",
        },
    }
    if preprocessor_config is not None:
        files["preprocessor_config.json"] = preprocessor_config
    for name, content in files.items():
        (model_dir / name).write_text(json.dumps(content), encoding="utf-8")
    return model_dir


@pytest.fixture
def model_dir(tmp_path: Path) -> Path:
    return _write_model_dir(tmp_path / "model", IMAGE_CONFIG)


def _colpali(model_dir: Path, tmp_path: Path, cls: type[ColPali] = ColPali, **kwargs: Any) -> Any:
    return cls(
        COLPALI,
        cache_dir=str(tmp_path / "cache"),
        specific_model_path=str(model_dir),
        **kwargs,
    )


def _facade(model_dir: Path, tmp_path: Path, **kwargs: Any) -> LateInteractionMultimodalEmbedding:
    return LateInteractionMultimodalEmbedding(
        COLPALI.lower(),
        cache_dir=str(tmp_path / "cache"),
        specific_model_path=str(model_dir),
        **kwargs,
    )


def _solid(color: tuple[int, int, int]) -> Image.Image:
    return Image.new("RGB", (8, 8), color)


def _saved(tmp_path: Path, name: str, color: tuple[int, int, int]) -> Path:
    path = tmp_path / name
    _solid(color).save(path, format="PNG")
    return path


def _patch_values(embedding: np.ndarray) -> float:
    """The single value a _StubSession image run spread over every patch of one image."""
    assert embedding.shape == (PATCHES, DIM)
    assert np.all(embedding == embedding[0, 0])
    return float(embedding[0, 0])


# --- ColPali: loading and preprocessor resolution ---


def test_colpali_reads_its_image_preprocessor_from_preprocessor_config(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _colpali(model_dir, tmp_path)

    assert model.preprocessor_spec == PreprocessorSpec(
        kind="image",
        config_file="preprocessor_config.json",
        image_size=(IMAGE_SIZE, IMAGE_SIZE),
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )
    assert stub_ort.created[0]["path"] == str(model_dir / "model.onnx")
    assert model.model_input_names == set(TEXT_IMAGE_INPUTS)
    assert model.static_batch_size is None
    assert model.special_token_to_id == {"<s>": BOS, "<pad>": PAD, "<unk>": VOCAB["<unk>"]}


def test_colpali_without_preprocessor_config_falls_back_to_default_image_preprocessing(
    tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model_dir = _write_model_dir(tmp_path / "bare", preprocessor_config=None)
    model = _colpali(model_dir, tmp_path)

    (embedding,) = model.embed_image([_solid(WHITE)])

    assert model.preprocessor_spec == PreprocessorSpec(kind="image")
    assert model.model.calls[-1]["pixel_values"].shape == (1, 3, 224, 224)
    assert embedding.shape == (PATCHES, DIM)


@pytest.mark.parametrize("source", ["text preprocessor_config.json", "declared text spec"])
def test_colpali_rejects_images_when_the_resolved_preprocessor_is_text(
    tmp_path: Path, stub_ort: _StubOrt, source: str
) -> None:
    if source == "declared text spec":
        model = _colpali(_write_model_dir(tmp_path / "m", None), tmp_path, lazy_load=True)
        model.declared_preprocessor = PreprocessorSpec(kind="text")
        model.load_onnx_model()
    else:
        text_config = {"processor_class": "PaliGemmaProcessor"}
        model = _colpali(_write_model_dir(tmp_path / "m", text_config), tmp_path)

    assert model.preprocessor_spec.kind == "text"
    with pytest.raises(
        ValueError,
        match=r"LateInteractionMultimodalEmbedding requires a preprocessor with kind='image' "
        r"or kind='none', got 'text'",
    ):
        list(model.embed_image([_solid(WHITE)]))


def test_colpali_declared_none_preprocessor_feeds_every_input_kind_as_raw_bytes(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _colpali(model_dir, tmp_path, lazy_load=True)
    model.declared_preprocessor = PreprocessorSpec(kind="none")
    model.load_onnx_model()
    str_path = tmp_path / "from-str.bin"
    str_path.write_bytes(b"read through a str path")
    path_path = tmp_path / "from-path.bin"
    path_path.write_bytes(b"read through a Path")
    image = _solid(RED)
    inputs = [
        b"bytes",
        bytearray(b"bytearray"),
        memoryview(b"memoryview"),
        str(str_path),
        path_path,
        image,
    ]

    embeddings = list(model.embed_image(inputs))

    sent = model.model.calls[-1]["pixel_values"]
    assert model.preprocessor_spec == PreprocessorSpec(kind="none")
    assert sent.dtype == object
    assert sent.tolist()[:5] == [
        b"bytes",
        b"bytearray",
        b"memoryview",
        b"read through a str path",
        b"read through a Path",
    ]
    png = sent[5]
    assert png.startswith(b"\x89PNG\r\n\x1a\n")
    with Image.open(io.BytesIO(png)) as decoded:
        assert decoded.size == image.size
        assert decoded.getpixel((0, 0)) == RED
    assert [_patch_values(embedding) for embedding in embeddings] == [
        float(len(item)) for item in sent
    ]


def test_colpali_raw_bytes_mode_wraps_a_single_bytes_input(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _colpali(model_dir, tmp_path, lazy_load=True)
    model.declared_preprocessor = PreprocessorSpec(kind="none")
    model.load_onnx_model()

    (embedding,) = model.embed_image(b"one image")

    assert model.model.calls[-1]["pixel_values"].tolist() == [b"one image"]
    assert _patch_values(embedding) == len(b"one image")


def test_colpali_raw_bytes_mode_rejects_unsupported_input_types(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _colpali(model_dir, tmp_path, lazy_load=True)
    model.declared_preprocessor = PreprocessorSpec(kind="none")
    model.load_onnx_model()

    with pytest.raises(TypeError, match="unsupported raw image input: int"):
        list(model.embed_image([42]))


def test_colpali_cannot_encode_images_before_a_preprocessor_is_resolved(
    model_dir: Path, tmp_path: Path
) -> None:
    model = _colpali(model_dir, tmp_path, lazy_load=True)

    assert model.preprocessor_spec is None
    with pytest.raises(ValueError, match="Image preprocessor is not initialized"):
        model._encode_images([_solid(WHITE)])


def test_colpali_onnx_calls_on_an_unloaded_model_ask_for_load_onnx_model(
    model_dir: Path, tmp_path: Path
) -> None:
    model = _colpali(model_dir, tmp_path, lazy_load=True)
    message = r"Model not loaded\. Please call load_onnx_model\(\) first\."

    with pytest.raises(ValueError, match=message):
        model.onnx_embed_text(["hello"])
    with pytest.raises(ValueError, match=message):
        model.onnx_embed_image([_solid(WHITE)])


@pytest.mark.parametrize(
    ("device_id", "device_ids", "expected"),
    [(3, [5, 6], 3), (None, [5, 6], 5), (None, None, None)],
)
def test_colpali_device_id_prefers_the_explicit_id_then_the_first_worker_device(
    model_dir: Path,
    tmp_path: Path,
    device_id: int | None,
    device_ids: list[int] | None,
    expected: int | None,
) -> None:
    model = _colpali(
        model_dir, tmp_path, lazy_load=True, device_id=device_id, device_ids=device_ids
    )

    assert model.device_id == expected
    assert model.model is None


def test_colpali_forwards_exposed_session_options_to_the_onnx_session(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _colpali(model_dir, tmp_path, enable_cpu_mem_arena=False)

    assert model._extra_session_options == {"enable_cpu_mem_arena": False}
    assert stub_ort.created[0]["sess_options"].enable_cpu_mem_arena is False


# --- ColPali: text ---


def test_colpali_embed_text_wraps_the_query_and_stamps_the_query_marker(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _colpali(model_dir, tmp_path)

    (embedding,) = model.embed_text("hello")

    sent = model.model.calls[-1]
    assert sent["input_ids"].tolist() == [HELLO_IDS]
    assert sent["attention_mask"].tolist() == [[1] * len(HELLO_IDS)]
    assert sent["pixel_values"].dtype == np.float32
    assert sent["pixel_values"].shape == (1, *ColPali.IMAGE_PLACEHOLDER_SIZE)
    assert not sent["pixel_values"].any()
    assert embedding.shape == (len(HELLO_IDS), DIM)
    assert embedding[:, 0].tolist() == HELLO_IDS


def test_colpali_embed_text_returns_one_padded_token_matrix_per_query_in_a_batch(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _colpali(model_dir, tmp_path)

    first, second = model.embed_text(["hello world", "hello"])

    sent = model.model.calls[-1]
    assert sent["input_ids"][:, :2].tolist() == [MARKER, MARKER]
    assert sent["attention_mask"].tolist() == [[1] * 15, [1] * 14 + [0]]
    assert first.shape == second.shape == (15, DIM)
    assert first[:4, 0].tolist() == [*MARKER, COLON, HELLO]
    assert first[4, 0] == WORLD


def test_colpali_text_input_omits_attention_mask_when_the_graph_does_not_declare_it(
    model_dir: Path, tmp_path: Path, install_ort: Any
) -> None:
    install_ort(input_names=("input_ids", "pixel_values"))
    model = _colpali(model_dir, tmp_path)

    output = model.onnx_embed_text(["hello"])

    assert set(model.model.calls[-1]) == {"input_ids", "pixel_values"}
    assert output.attention_mask.tolist() == [[1] * len(HELLO_IDS)]
    assert output.input_ids.tolist() == [HELLO_IDS]


def test_colpali_text_input_adds_zero_token_type_ids_when_the_graph_declares_them(
    model_dir: Path, tmp_path: Path, install_ort: Any
) -> None:
    install_ort(input_names=(*TEXT_IMAGE_INPUTS, "token_type_ids"))
    model = _colpali(model_dir, tmp_path)

    list(model.embed_text("hello"))

    token_type_ids = model.model.calls[-1]["token_type_ids"]
    assert token_type_ids.dtype == np.int64
    assert token_type_ids.tolist() == [[0] * len(HELLO_IDS)]


def test_colpali_embed_text_loads_a_lazy_model_and_accepts_any_iterable(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _colpali(model_dir, tmp_path, lazy_load=True)
    assert not stub_ort.created

    embeddings = list(model.embed_text(query for query in ["hello", "world", "hello"]))

    assert len(stub_ort.created) == 1
    assert [embedding[3, 0] for embedding in embeddings] == [HELLO, WORLD, HELLO]


def test_colpali_embed_text_stays_in_process_without_parallel_for_large_lists(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _colpali(model_dir, tmp_path)

    embeddings = list(model.embed_text(["hello", "world", "hello"], batch_size=2))

    assert [call["input_ids"].shape[0] for call in model.model.calls] == [2, 1]
    assert len(embeddings) == 3


# --- ColPali: images ---


def test_colpali_embed_image_normalizes_pixels_and_reshapes_output_into_patch_rows(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _colpali(model_dir, tmp_path)

    embeddings = list(model.embed_image([_solid(WHITE), _solid(BLACK), _solid(RED)]))

    sent = model.model.calls[-1]
    assert sent["pixel_values"].dtype == np.float32
    assert sent["pixel_values"].shape == (3, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert sent["input_ids"].tolist() == [ColPali.EMPTY_TEXT_PLACEHOLDER.tolist()] * 3
    assert sent["attention_mask"].tolist() == [[1] * 1030] * 3
    np.testing.assert_allclose(
        [_patch_values(embedding) for embedding in embeddings],
        [NORMALIZED[WHITE], NORMALIZED[BLACK], NORMALIZED[RED]],
        atol=1e-6,
    )


@pytest.mark.parametrize("kind", ["PIL image", "str path", "Path"])
def test_colpali_embed_image_wraps_a_single_image_into_a_batch_of_one(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt, kind: str
) -> None:
    model = _colpali(model_dir, tmp_path)
    path = _saved(tmp_path, "black.png", BLACK)
    image = {"PIL image": _solid(BLACK), "str path": str(path), "Path": path}[kind]

    (embedding,) = model.embed_image(image)

    assert model.model.calls[-1]["pixel_values"].shape[0] == 1
    assert _patch_values(embedding) == pytest.approx(NORMALIZED[BLACK], abs=1e-6)


def test_colpali_embed_image_loads_a_lazy_model_and_accepts_any_iterable(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _colpali(model_dir, tmp_path, lazy_load=True)

    embeddings = list(model.embed_image(_solid(color) for color in (WHITE, BLACK)))

    assert len(stub_ort.created) == 1
    np.testing.assert_allclose(
        [_patch_values(embedding) for embedding in embeddings],
        [NORMALIZED[WHITE], NORMALIZED[BLACK]],
        atol=1e-6,
    )


def test_colpali_static_batch_graph_caps_text_and_image_batches(
    model_dir: Path, tmp_path: Path, install_ort: Any
) -> None:
    install_ort(batch_dim=2)
    model = _colpali(model_dir, tmp_path)

    texts = list(model.embed_text(["hello", "world", "hello"]))
    images = list(model.embed_image([_solid(WHITE), _solid(BLACK), _solid(RED)]))

    assert model.static_batch_size == 2
    assert [len(call["input_ids"]) for call in model.model.calls] == [2, 1, 2, 1]
    assert len(texts) == len(images) == 3


def test_colpali_casts_float16_graph_output_to_float32(
    model_dir: Path, tmp_path: Path, install_ort: Any
) -> None:
    install_ort(output_dtype=np.float16)
    model = _colpali(model_dir, tmp_path)

    (text,) = model.embed_text("hello")
    (image,) = model.embed_image(_solid(WHITE))

    assert text.dtype == np.float32
    assert image.dtype == np.float32
    assert _patch_values(image) == pytest.approx(NORMALIZED[WHITE], abs=1e-3)


# --- ColPali: token counting ---


@pytest.mark.parametrize(
    ("texts", "kwargs", "expected"),
    [
        ("hello", {}, 1),
        (["hello world", "hello"], {}, 3),
        (["hello world", "hello"], {"include_extension": True}, 3 + 2 * EXTENSION),
        (
            ["hello world", "hello"],
            {"include_extension": True, "batch_size": 1},
            3 + 2 * EXTENSION,
        ),
    ],
)
def test_colpali_token_count_counts_attended_tokens_and_optionally_the_query_wrapper(
    model_dir: Path,
    tmp_path: Path,
    stub_ort: _StubOrt,
    texts: str | list[str],
    kwargs: dict[str, Any],
    expected: int,
) -> None:
    model = _colpali(model_dir, tmp_path, lazy_load=True)

    assert model.token_count(texts, **kwargs) == expected
    assert len(stub_ort.created) == 1  # counting loads the lazy model for its tokenizer


def test_colpali_token_count_reuses_an_already_loaded_model(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    model = _colpali(model_dir, tmp_path)

    assert model.token_count("hello world") == 2
    assert len(stub_ort.created) == 1


# --- ColPali: workers and the parallel path ---


def test_colpali_text_worker_builds_a_single_threaded_model_and_embeds_batches(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    assert ColPali._get_text_worker_class() is ColPaliTextEmbeddingWorker

    worker = ColPaliTextEmbeddingWorker(
        COLPALI, str(tmp_path / "cache"), specific_model_path=str(model_dir)
    )
    ((index, output),) = worker.process([(4, ["hello"])])

    assert isinstance(worker.model, ColPali)
    assert worker.model.threads == 1
    assert stub_ort.created[0]["sess_options"].intra_op_num_threads == 1
    assert index == 4
    assert output.input_ids is not None
    assert output.input_ids.tolist() == [HELLO_IDS]


def test_colpali_image_worker_builds_a_single_threaded_model_and_embeds_batches(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    assert ColPali._get_image_worker_class() is ColPaliImageEmbeddingWorker

    worker = ColPaliImageEmbeddingWorker(
        COLPALI, str(tmp_path / "cache"), specific_model_path=str(model_dir)
    )
    ((index, output),) = worker.process([(9, [_solid(WHITE)])])

    assert isinstance(worker.model, ColPali)
    assert worker.model.threads == 1
    assert index == 9
    assert output.model_output.shape == (1, PATCHES * DIM)
    np.testing.assert_allclose(output.model_output, NORMALIZED[WHITE], atol=1e-6)


def test_colpali_parallel_zero_hands_text_and_images_to_a_worker_pool(
    model_dir: Path, tmp_path: Path
) -> None:
    # The pool deep-copies the worker kwargs before it starts any process, so a generator
    # kwarg fails there; the in-process path would ignore it. No worker process is spawned.
    # The local model directory and local_files_only still go to the workers: if a pool ever
    # did start, it must load the tmp_path model instead of downloading the real 5.8 GB one.
    model = _colpali(model_dir, tmp_path, cls=_SpawnableColPali, lazy_load=True)
    common = {
        "model_name": COLPALI,
        "cache_dir": model.cache_dir,
        "batch_size": 1,
        "parallel": 0,
        "specific_model_path": str(model_dir),
        "local_files_only": True,
    }

    with pytest.raises(TypeError, match="cannot pickle 'generator' object"):
        list(
            model._embed_documents(
                documents=["hello", "world"], probe=(n for n in range(1)), **common
            )
        )
    with pytest.raises(TypeError, match="cannot pickle 'generator' object"):
        list(model._embed_images(images=[b"a", b"b"], probe=(n for n in range(1)), **common))
    assert model.model is None


def test_colpali_parallel_text_embedding_in_a_spawned_worker_matches_in_process_output(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    queries = ["hello", "hello world"]
    expected = list(_colpali(model_dir, tmp_path).embed_text(queries, batch_size=1))
    model = _colpali(model_dir, tmp_path, cls=_SpawnableColPali, lazy_load=True)

    embeddings = list(model.embed_text(queries, batch_size=1, parallel=1))

    assert model.model is None
    assert len(stub_ort.created) == 1  # only the in-process reference model
    assert [embedding.shape for embedding in embeddings] == [(14, DIM), (15, DIM)]
    for embedding, reference in zip(embeddings, expected, strict=True):
        np.testing.assert_array_equal(embedding, reference)


def test_colpali_parallel_image_embedding_in_a_spawned_worker_matches_in_process_output(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    paths = [str(_saved(tmp_path, "white.png", WHITE)), str(_saved(tmp_path, "black.png", BLACK))]
    model = _colpali(model_dir, tmp_path, cls=_SpawnableColPali, lazy_load=True)

    embeddings = list(model.embed_image(paths, batch_size=1, parallel=1))

    assert model.model is None
    assert not stub_ort.created
    np.testing.assert_allclose(
        [_patch_values(embedding) for embedding in embeddings],
        [NORMALIZED[WHITE], NORMALIZED[BLACK]],
        atol=1e-6,
    )


# --- LateInteractionMultimodalEmbedding facade ---


def test_multimodal_facade_lists_colpali_descriptions_as_dicts() -> None:
    assert LateInteractionMultimodalEmbedding.list_supported_models() == [
        asdict(description) for description in supported_colpali_models
    ]


def test_multimodal_facade_resolves_colpali_case_insensitively_and_delegates(
    model_dir: Path, tmp_path: Path, stub_ort: _StubOrt
) -> None:
    facade = _facade(model_dir, tmp_path)

    (text,) = facade.embed_text("hello")
    (image,) = facade.embed_image([_solid(WHITE)])

    assert isinstance(facade.model, ColPali)
    assert facade.model.model_description.model == COLPALI
    assert text[:, 0].tolist() == HELLO_IDS
    assert _patch_values(image) == pytest.approx(NORMALIZED[WHITE], abs=1e-6)
    assert facade.token_count("hello") == 1
    assert facade.token_count(["hello"], batch_size=1, include_extension=True) == 1 + EXTENSION


def test_multimodal_facade_rejects_unregistered_models_before_loading_anything(
    tmp_path: Path, stub_ort: _StubOrt
) -> None:
    with pytest.raises(
        ValueError,
        match=r"^Model acme/none is not supported in LateInteractionMultimodalEmbedding\. "
        r"Please check the supported models using "
        r"`LateInteractionMultimodalEmbedding\.list_supported_models\(\)`$",
    ):
        LateInteractionMultimodalEmbedding("acme/none", cache_dir=str(tmp_path / "cache"))
    assert not stub_ort.created


def test_multimodal_facade_embedding_size_is_looked_up_once_then_cached(
    model_dir: Path, tmp_path: Path
) -> None:
    facade = _facade(model_dir, tmp_path, lazy_load=True)
    assert facade._embedding_size is None

    assert facade.embedding_size == DIM
    assert facade._embedding_size == DIM
    assert facade.embedding_size == DIM


def test_multimodal_get_embedding_size_matches_case_insensitively_or_lists_known_models() -> None:
    assert LateInteractionMultimodalEmbedding.get_embedding_size(COLPALI.upper()) == DIM
    with pytest.raises(
        ValueError,
        match=r"Embedding size for model acme/none was None\. "
        r"Available model names: \['Qdrant/colpali-v1\.3-fp16'\]",
    ):
        LateInteractionMultimodalEmbedding.get_embedding_size("acme/none")


# --- LateInteractionMultimodalEmbeddingBase ---


def test_multimodal_base_stores_settings_and_leaves_model_methods_unimplemented() -> None:
    base = LateInteractionMultimodalEmbeddingBase(
        "acme/model", cache_dir="cache", threads=2, local_files_only=True
    )

    assert (base.model_name, base.cache_dir, base.threads) == ("acme/model", "cache", 2)
    assert base._local_files_only is True
    assert base._embedding_size is None
    with pytest.raises(NotImplementedError, match="^$"):
        base.embed_text(["hello"])
    with pytest.raises(NotImplementedError, match="^$"):
        base.embed_image([b"image"])
    with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
        LateInteractionMultimodalEmbeddingBase.get_embedding_size("acme/model")
    with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
        _ = base.embedding_size
    with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
        base.token_count("hello")


# --- OnnxMultimodalModel ---


def test_onnx_multimodal_model_passes_inputs_through_and_leaves_model_hooks_unimplemented() -> (
    None
):
    runner: OnnxMultimodalModel[np.ndarray] = OnnxMultimodalModel()
    onnx_input: dict[str, NumpyArray] = {"input_ids": np.zeros((1, 2), dtype=np.int64)}
    output = OnnxOutputContext(model_output=np.zeros((1, 2), dtype=np.float32))

    assert runner.model is None
    assert runner.preprocessor_spec is None
    assert runner._preprocess_onnx_text_input(onnx_input) is onnx_input
    assert runner._preprocess_onnx_image_input(onnx_input) is onnx_input
    for hook in (
        OnnxMultimodalModel._get_text_worker_class,
        OnnxMultimodalModel._get_image_worker_class,
        runner.load_onnx_model,
        lambda: runner._post_process_onnx_text_output(output),
        lambda: runner._post_process_onnx_image_output(output),
    ):
        with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
            hook()


def test_onnx_multimodal_model_tokenize_encodes_documents_without_any_query_wrapper(
    model_dir: Path,
) -> None:
    runner: OnnxMultimodalModel[np.ndarray] = OnnxMultimodalModel()
    runner.tokenizer, _ = load_tokenizer(model_dir=model_dir)

    encodings = runner.tokenize(["hello world", "hello"])

    assert [encoding.ids for encoding in encodings] == [[HELLO, WORLD], [HELLO, PAD]]
    assert [encoding.attention_mask for encoding in encodings] == [[1, 1], [1, 0]]
