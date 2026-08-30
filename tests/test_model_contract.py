import json
from pathlib import Path

import pytest

from fastretrieval.contract import ModelContract, PreprocessorSpec, write_manifest


def _payload(model_id: str = "acme/tiny-e5") -> dict:
    return {
        "schema_version": 1,
        "model_id": model_id,
        "source": model_id,
        "model_family": "e5",
        "task": "dense",
        "modality": "text",
        "output_dim": 384,
        "output_shape": [384],
        "pooling": "MEAN",
        "normalization": True,
        "max_seq_len": 512,
        "preprocessor": {"kind": "text"},
        "tokenizer_files": ["tokenizer.json"],
        "artifact_formats": ["onnx"],
        "quantization": None,
        "exporter_version": "fixture",
    }


def _write_manifest(tmp_path: Path, payload: dict) -> Path:
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_non_qwen_manifest_roundtrips_and_adapts_to_dense_runtime(tmp_path: Path):
    contract = ModelContract.from_manifest(_write_manifest(tmp_path, _payload()))

    assert contract.model_id == "acme/tiny-e5"
    assert contract.model_family == "e5"
    assert contract.preprocessor == PreprocessorSpec(kind="text")
    assert contract.to_custom_model_spec_kwargs() == {
        "model_id": "acme/tiny-e5",
        "hf": "acme/tiny-e5",
        "model_file": "onnx/model.onnx",
        "dim": 384,
        "pooling": "MEAN",
        "normalization": True,
        "max_seq_len": 512,
        "preprocessor": PreprocessorSpec(kind="text"),
    }


def test_qwen_reference_uses_the_same_manifest_code_path(tmp_path: Path):
    payload = _payload("n24q02m/Qwen3-Embedding-0.6B")
    payload["model_family"] = "qwen3"
    contract = ModelContract.from_manifest(_write_manifest(tmp_path, payload))

    assert contract.model_id == "n24q02m/Qwen3-Embedding-0.6B"
    assert contract.to_custom_model_spec_kwargs(hf="mirror/qwen3")["hf"] == "mirror/qwen3"


def test_manifest_rejects_mismatched_embedding_dimension(tmp_path: Path):
    payload = _payload("acme/mismatched-dimension")
    payload["output_shape"] = [385]

    with pytest.raises(ValueError, match="output_dim disagrees with output_shape"):
        ModelContract.from_manifest(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("task", None, "task"),
        ("modality", None, "modality"),
        ("output_dim", None, "output_dim"),
        ("output_shape", None, "output_shape"),
        ("artifact_formats", ["torchscript"], "artifact_formats"),
    ],
)
def test_manifest_rejects_missing_or_unsupported_fields(
    tmp_path: Path, field: str, value: object, message: str
):
    payload = _payload("acme/bad")
    if field == "output_dim":
        payload["output_shape"] = None
    if field == "output_shape":
        payload["output_dim"] = None
    payload[field] = value

    with pytest.raises(ValueError, match=f"acme/bad.*{message}"):
        ModelContract.from_manifest(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize(
    "tokenizer_path", ["../outside", r"..\outside", "/absolute/file", r"C:\outside"]
)
def test_manifest_rejects_tokenizer_paths_outside_artifact_root(
    tmp_path: Path, tokenizer_path: str
):
    payload = _payload("acme/path-traversal")
    payload["tokenizer_files"] = [tokenizer_path]

    with pytest.raises(ValueError, match="safe relative paths"):
        ModelContract.from_manifest(_write_manifest(tmp_path, payload))


def test_write_manifest_preserves_only_supplied_metadata(tmp_path: Path):
    contract = ModelContract.from_manifest(_write_manifest(tmp_path, _payload()))
    path = tmp_path / "written.json"
    write_manifest(
        path,
        contract,
        {"license": "Apache-2.0", "quantization": "int8", "exporter_version": "0.2"},
    )

    written = json.loads(path.read_text(encoding="utf-8"))
    assert written["license"] == "Apache-2.0"
    assert written["quantization"] == "int8"
    assert written["exporter_version"] == "0.2"
    assert "provenance" not in written
    assert list(written) == sorted(written)
