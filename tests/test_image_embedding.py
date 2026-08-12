import numpy as np
import pytest

from fastretrieval.contract import PreprocessorSpec

PIL = pytest.importorskip("PIL.Image")


def _solid(color: tuple[int, int, int], size: int = 64):
    return PIL.new("RGB", (size, size), color)


def test_preprocess_resizes_and_normalizes():
    from fastretrieval.preprocessing.image import preprocess_images

    spec = PreprocessorSpec(
        kind="image",
        image_size=(224, 224),
        image_mean=(0.5, 0.5, 0.5),
        image_std=(0.5, 0.5, 0.5),
    )
    out = preprocess_images([_solid((255, 255, 255))], spec)
    assert out.shape == (1, 3, 224, 224)
    assert out.dtype == np.float32
    assert np.allclose(out, 1.0, atol=1e-4)


def test_preprocess_uses_defaults_when_spec_is_bare():
    from fastretrieval.preprocessing.image import preprocess_images

    out = preprocess_images([_solid((0, 0, 0))], PreprocessorSpec(kind="image"))
    assert out.shape[1] == 3
    assert out.dtype == np.float32


def test_preprocess_rejects_text_spec():
    from fastretrieval.preprocessing.image import preprocess_images

    with pytest.raises(ValueError, match="kind='image'"):
        preprocess_images([_solid((0, 0, 0))], PreprocessorSpec(kind="text"))


def test_onnx_image_runner_uses_contract_without_loading_text_tokenizer(tmp_path, monkeypatch):
    from fastretrieval.models.image.onnx_image_model import OnnxImageModel

    class _FakeInput:
        name = "pixel_values"
        shape = ["batch", 3, 224, 224]

    class _FakeSession:
        def get_inputs(self):
            return [_FakeInput()]

        def get_outputs(self):
            return []

    class _ProbeImageModel(OnnxImageModel[np.ndarray]):
        @classmethod
        def _get_worker_class(cls):
            raise NotImplementedError

        def _post_process_onnx_output(self, output, **kwargs):
            return ()

    runner = _ProbeImageModel()

    def _fake_instantiate(**kwargs):
        return _FakeSession(), ["pixel_values"]

    monkeypatch.setattr(runner, "_instantiate_onnx_session", _fake_instantiate)
    runner._load_onnx_model(tmp_path, "model.onnx")

    assert runner.tokenizer is None
    assert runner.preprocessor_spec == PreprocessorSpec(kind="image")
    encoded = np.zeros((1, 3, 224, 224), dtype=np.float32)
    onnx_input = runner._build_onnx_input(encoded)
    assert set(onnx_input) == {"pixel_values"}
    np.testing.assert_array_equal(onnx_input["pixel_values"], encoded)


def test_self_contained_image_graph_keeps_raw_bytes():
    from fastretrieval.models.image.onnx_image_model import OnnxImageModel

    class _ProbeImageModel(OnnxImageModel[np.ndarray]):
        @classmethod
        def _get_worker_class(cls):
            raise NotImplementedError

        def _post_process_onnx_output(self, output, **kwargs):
            return ()

    runner = _ProbeImageModel()
    runner.preprocessor_spec = PreprocessorSpec(kind="none")
    encoded = runner._encode_images([b"raw-image"])

    assert encoded.dtype == object
    assert encoded.tolist() == [b"raw-image"]
