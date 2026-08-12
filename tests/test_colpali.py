import pytest

from fastretrieval.models.late_interaction_multimodal import (
    LateInteractionMultimodalEmbedding,
)


def test_list_supported_models_is_not_empty():
    models = LateInteractionMultimodalEmbedding.list_supported_models()
    assert models
    assert all("model" in m and "dim" in m for m in models)
    assert any("qwen" not in m["model"].lower() for m in models)


def test_unknown_model_raises():
    with pytest.raises(ValueError, match="not supported"):
        LateInteractionMultimodalEmbedding(model_name="acme/does-not-exist")


def test_image_path_requires_the_image_extra(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("PIL"):
            raise ImportError("no pillow")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    from fastretrieval.contract import PreprocessorSpec
    from fastretrieval.preprocessing.image import preprocess_images

    with pytest.raises(ImportError, match="fastretrieval\\[image\\]"):
        preprocess_images(["whatever.png"], PreprocessorSpec(kind="image"))


def test_multimodal_self_contained_graph_keeps_raw_bytes():
    import numpy as np

    from fastretrieval.contract import PreprocessorSpec
    from fastretrieval.models.late_interaction_multimodal.onnx_multimodal_model import (
        OnnxMultimodalModel,
    )

    class _FakeInput:
        name = "pixel_values"

    class _FakeSession:
        def __init__(self):
            self.inputs = None

        def get_inputs(self):
            return [_FakeInput()]

        def run(self, _output_names, inputs):
            self.inputs = inputs
            return [np.zeros((1, 1), dtype=np.float32)]

    class _ProbeMultimodalModel(OnnxMultimodalModel[np.ndarray]):
        @classmethod
        def _get_text_worker_class(cls):
            raise NotImplementedError

        @classmethod
        def _get_image_worker_class(cls):
            raise NotImplementedError

        def _post_process_onnx_text_output(self, output):
            return ()

        def _post_process_onnx_image_output(self, output):
            return ()

    runner = _ProbeMultimodalModel()
    session = _FakeSession()
    runner.model = session
    runner.preprocessor_spec = PreprocessorSpec(kind="none")

    runner.onnx_embed_image([b"raw-image"])

    assert session.inputs["pixel_values"].dtype == object
    assert session.inputs["pixel_values"].tolist() == [b"raw-image"]


def test_colpali_lazy_load_initializes_the_onnx_runner(tmp_path, monkeypatch):
    from fastretrieval.models.late_interaction_multimodal.colpali import ColPali

    monkeypatch.setattr(ColPali, "download_model", lambda *args, **kwargs: str(tmp_path))

    model = ColPali(
        model_name="Qdrant/colpali-v1.3-fp16",
        cache_dir=str(tmp_path),
        lazy_load=True,
    )

    assert model.model is None
