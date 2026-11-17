import json
import sys
from pathlib import Path
from typing import cast

import pytest

from fastretrieval.contract import PreprocessorSpec
from fastretrieval.preprocessing.graph import graph_is_self_contained, resolve_preprocessor


def _write_config(directory: Path, payload: dict) -> None:
    (directory / "preprocessor_config.json").write_text(json.dumps(payload), encoding="utf-8")


def _write_graph(path: Path, *extra_domains: str) -> Path:
    import onnx
    from onnx import TensorProto, helper

    graph = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["y"])],
        "probe",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
    )
    opsets = [helper.make_opsetid("", 17)]
    opsets += [helper.make_opsetid(domain, 1) for domain in extra_domains]
    path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(helper.make_model(graph, opset_imports=opsets), str(path))
    return path


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


@pytest.mark.parametrize("domain", ["ai.onnx.contrib", "com.microsoft.extensions"])
def test_graph_with_an_extension_opset_is_self_contained(tmp_path: Path, domain: str):
    assert graph_is_self_contained(_write_graph(tmp_path / "model.onnx", "com.example", domain))


def test_graph_with_only_standard_opsets_is_not_self_contained(tmp_path: Path):
    assert not graph_is_self_contained(_write_graph(tmp_path / "model.onnx", "com.example"))


def test_unreadable_or_missing_graph_is_not_self_contained(tmp_path: Path):
    garbage = tmp_path / "model.onnx"
    garbage.write_bytes(b"\xff\xfe not a protobuf")

    assert not graph_is_self_contained(garbage)
    assert not graph_is_self_contained(tmp_path / "missing.onnx")


def test_graph_check_is_false_when_onnx_is_not_installed(tmp_path: Path, monkeypatch):
    path = _write_graph(tmp_path / "model.onnx", "ai.onnx.contrib")
    # Optional-dependency boundary: hide onnx the way an install without it would.
    monkeypatch.setitem(sys.modules, "onnx", None)

    assert not graph_is_self_contained(path)


def test_real_self_contained_graph_overrides_a_declared_spec(tmp_path: Path):
    _write_graph(tmp_path / "onnx" / "model.onnx", "ai.onnx.contrib")
    declared = PreprocessorSpec(kind="image", image_size=(336, 336))

    got = resolve_preprocessor(tmp_path, "onnx/model.onnx", declared)

    assert got == PreprocessorSpec(kind="none")


def test_text_only_config_file_resolves_to_a_text_spec(tmp_path: Path):
    _write_graph(tmp_path / "model.onnx")
    _write_config(tmp_path, {"do_lower_case": True})

    # resolve_preprocessor coerces a str directory at runtime; keep exercising that input.
    got = resolve_preprocessor(cast(Path, str(tmp_path)), "model.onnx", None)

    assert got == PreprocessorSpec(kind="text", config_file="preprocessor_config.json")
