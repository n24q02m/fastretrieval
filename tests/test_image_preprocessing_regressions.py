"""Regression tests for the image preprocessing defects inherited from fastembed 0.8.0.

The ONNX session is the only thing faked here: ``_instantiate_onnx_session`` is
replaced by a session whose output is the per-channel mean of the pixels it is
fed, so no model is downloaded and onnxruntime is not needed. Everything between
the caller and that boundary (``embed``, lazy loading, preprocessing, Pillow
decoding and resizing) runs for real.
"""

import io

import numpy as np
import pytest

from fastretrieval.contract import PreprocessorSpec
from fastretrieval.models.image.onnx_embedding import OnnxImageEmbedding
from fastretrieval.preprocessing.image import preprocess_images

# Pillow ships in the optional [image] extra. The integration job installs only the
# dev group and still collects every module, so a hard import fails it at collection.
# The transform operators import Pillow at module level, so the tests import them
# only after this guard.
Image = pytest.importorskip("PIL.Image", reason="Pillow belongs to the optional [image] extra")

RED = (255, 0, 0)
# Identity normalization: the tensor fed to the graph is the pixels scaled to [0, 1].
SPEC = PreprocessorSpec(
    kind="image", image_size=(2, 2), image_mean=(0.0, 0.0, 0.0), image_std=(1.0, 1.0, 1.0)
)


class _Input:
    name = "pixel_values"
    shape = ["batch", 3, 2, 2]


class _ChannelMeanSession:
    def get_inputs(self):
        return [_Input()]

    def get_outputs(self):
        return []

    def run(self, output_names, feed):
        return [feed["pixel_values"].mean(axis=(2, 3))]


@pytest.fixture
def opened_sessions(monkeypatch):
    sessions = []

    def _instantiate(self, model_path, config=None):
        sessions.append(_ChannelMeanSession())
        return sessions[-1], ["pixel_values"]

    monkeypatch.setattr(OnnxImageEmbedding, "_instantiate_onnx_session", _instantiate)
    return sessions


def _embedding(tmp_path, **kwargs):
    model_dir = tmp_path / "model"
    model_dir.mkdir(exist_ok=True)
    (model_dir / "model.onnx").write_bytes(b"not a real graph; the session is faked")
    return OnnxImageEmbedding(
        model_name="Qdrant/clip-ViT-B-32-vision",
        cache_dir=str(tmp_path / "cache"),
        specific_model_path=str(model_dir),
        preprocessor=SPEC,
        **kwargs,
    )


def _png(color, size=(4, 2)):
    output = io.BytesIO()
    Image.new("RGB", size, color).save(output, format="PNG")
    return output.getvalue()


@pytest.mark.parametrize("processor", ["CLIPImageProcessor", "SiglipImageProcessor"])
def test_height_width_resize_keeps_height_on_the_first_spatial_axis(processor):
    from fastretrieval.models.image.transform.operators import Compose

    config = {
        "image_processor_type": processor,
        "do_resize": True,
        "size": {"height": 2, "width": 4},
        "do_rescale": False,
    }

    (out,) = Compose.from_config(config)([Image.new("RGB", (8, 8))])

    assert out.shape == (3, 2, 4)


def test_lazy_image_embedding_opens_the_session_on_the_first_embed(opened_sessions, tmp_path):
    emb = _embedding(tmp_path, lazy_load=True)

    assert emb.model is None
    assert opened_sessions == []

    result = list(emb.embed(Image.new("RGB", (4, 4), RED)))

    assert len(opened_sessions) == 1
    np.testing.assert_allclose(result, [[1.0, 0.0, 0.0]], atol=1e-6)


@pytest.mark.parametrize("wrap", [bytes, bytearray, memoryview])
def test_preprocess_images_decodes_encoded_image_bytes(wrap):
    out = preprocess_images([wrap(_png(RED))], SPEC)

    assert out.shape == (1, 3, 2, 2)
    np.testing.assert_allclose(out[0, :, 0, 0], [1.0, 0.0, 0.0], atol=1e-6)


def test_image_embedding_embeds_a_single_encoded_image(opened_sessions, tmp_path):
    emb = _embedding(tmp_path)

    result = list(emb.embed(_png(RED)))

    np.testing.assert_allclose(result, [[1.0, 0.0, 0.0]], atol=1e-6)


def test_interpolation_resolver_maps_nearest():
    from fastretrieval.models.image.transform.operators import Compose

    assert Compose._interpolation_resolver("Nearest") is Image.Resampling.NEAREST


def test_jina_config_keeps_nearest_given_by_name():
    from fastretrieval.models.image.transform.operators import Compose, Resize

    config = {
        "image_processor_type": "JinaCLIPImageProcessor",
        "size": 4,
        "interpolation": "nearest",
    }

    resize = Compose.from_config(config).transforms[1]

    assert isinstance(resize, Resize)
    assert resize.resample == Image.Resampling.NEAREST


def test_jina_config_keeps_nearest_given_as_an_integer():
    from fastretrieval.models.image.transform.operators import Compose, Resize

    config = {
        "image_processor_type": "JinaCLIPImageProcessor",
        "size": 4,
        "interpolation": int(Image.Resampling.NEAREST),
    }

    resize = Compose.from_config(config).transforms[1]

    assert isinstance(resize, Resize)
    assert resize.resample == Image.Resampling.NEAREST


def test_convnext_config_without_a_size_is_rejected_with_a_value_error():
    from fastretrieval.models.image.transform.operators import Compose

    with pytest.raises(ValueError, match="must contain 'shortest_edge' key"):
        Compose.from_config({"image_processor_type": "ConvNextFeatureExtractor"})
