import json
from pathlib import Path

from fastretrieval.contract import PreprocessorSpec
from fastretrieval.preprocessing.graph import resolve_preprocessor


def _write_config(directory: Path, payload: dict) -> None:
    (directory / "preprocessor_config.json").write_text(json.dumps(payload), encoding="utf-8")


def test_declared_spec_wins_over_config_file(tmp_path: Path):
    _write_config(tmp_path, {"image_processor_type": "CLIPImageProcessor", "size": 224})
    declared = PreprocessorSpec(kind="image", image_size=(336, 336))
    got = resolve_preprocessor(tmp_path, "onnx/model.onnx", declared)
    assert got.image_size == (336, 336)


def test_config_file_used_when_nothing_declared(tmp_path: Path):
    _write_config(tmp_path, {"image_processor_type": "CLIPImageProcessor", "size": 224})
    got = resolve_preprocessor(tmp_path, "onnx/model.onnx", None)
    assert got.kind == "image"
    assert got.image_size == (224, 224)


def test_falls_back_to_text_when_nothing_available(tmp_path: Path):
    got = resolve_preprocessor(tmp_path, "onnx/model.onnx", None)
    assert got.kind == "text"


def test_self_contained_graph_short_circuits_to_none(tmp_path: Path, monkeypatch):
    _write_config(tmp_path, {"image_processor_type": "CLIPImageProcessor", "size": 224})
    monkeypatch.setattr(
        "fastretrieval.preprocessing.graph.graph_is_self_contained", lambda _p: True
    )
    got = resolve_preprocessor(tmp_path, "onnx/model.onnx", None)
    assert got.kind == "none"
