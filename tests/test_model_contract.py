import json
from pathlib import Path

import pytest

from fastretrieval.common.model_description import PoolingType
from fastretrieval.contract import (
    ModelContract,
    PreprocessorSpec,
    load_manifest,
    write_manifest,
)


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


def _contract_kwargs(**overrides: object) -> dict:
    kwargs: dict = {
        "model_id": "acme/tiny-e5",
        "source": "acme/tiny-e5",
        "task": "dense",
        "modality": "text",
        "model_family": "e5",
        "output_dim": 384,
        "output_shape": (384,),
        "pooling": "MEAN",
        "normalization": True,
        "max_seq_len": 512,
        "preprocessor": PreprocessorSpec(),
        "artifact_formats": ("onnx",),
        "tokenizer_files": ("tokenizer.json",),
    }
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"model_id": "   "}, "model_id must not be empty"),
        ({"model_id": 7}, "model_id must not be empty"),
        ({"source": ""}, "source must not be empty"),
        ({"model_family": " "}, "model_family must not be empty"),
        ({"task": "classify"}, "unsupported task 'classify'"),
        ({"task": None}, "unsupported task None"),
        ({"modality": "smell"}, "unsupported modality 'smell'"),
        ({"pooling": "median"}, "unsupported pooling 'median'"),
        ({"normalization": "yes"}, "normalization must be boolean"),
        ({"output_dim": None, "output_shape": None}, "output_dim or output_shape is required"),
        ({"output_dim": 0}, "output_dim must be positive"),
        ({"output_dim": True}, "output_dim must be positive"),
        ({"output_dim": "384"}, "output_dim must be positive"),
        ({"output_shape": "384"}, "output_shape must be a tuple or list"),
        ({"output_shape": ()}, "output_shape must not be empty"),
        ({"output_shape": (0,)}, "output_shape values must be positive"),
        ({"output_shape": (True,)}, "output_shape values must be positive"),
        ({"output_shape": ("384",)}, "output_shape values must be positive"),
        ({"max_seq_len": 0}, "max_seq_len must be positive"),
        ({"max_seq_len": False}, "max_seq_len must be positive"),
        ({"preprocessor": {"kind": "text"}}, "preprocessor must be a PreprocessorSpec"),
        ({"artifact_formats": "onnx"}, "artifact_formats must be a tuple or list"),
        ({"artifact_formats": ()}, "artifact_formats is required"),
        ({"artifact_formats": ("onnx", "onnx")}, "artifact_formats must not contain duplicates"),
        ({"tokenizer_files": "tokenizer.json"}, "tokenizer_files must be a tuple or list"),
        ({"tokenizer_files": ("",)}, "tokenizer_files must contain non-empty strings"),
        ({"tokenizer_files": (3,)}, "tokenizer_files must contain non-empty strings"),
        ({"tokenizer_files": ("tok\x00.json",)}, "safe relative paths"),
        ({"tokenizer_files": ("a.json", "a.json")}, "tokenizer_files must not contain duplicates"),
    ],
)
def test_contract_rejects_invalid_field_values(overrides: dict, message: str):
    with pytest.raises(ValueError, match=message):
        ModelContract(**_contract_kwargs(**overrides))


def test_contract_normalizes_pooling_and_sequence_fields():
    contract = ModelContract(
        **_contract_kwargs(
            pooling="cls",
            output_shape=[384],
            artifact_formats=["onnx", "gguf"],
            tokenizer_files=["tokenizer.json", "vocab/merges.txt"],
        )
    )

    assert contract.pooling is PoolingType.CLS
    assert contract.output_shape == (384,)
    assert contract.artifact_formats == ("onnx", "gguf")
    assert contract.tokenizer_files == ("tokenizer.json", "vocab/merges.txt")


def test_contract_keeps_pooling_enum_and_absent_output_shape():
    contract = ModelContract(
        **_contract_kwargs(pooling=PoolingType.LAST_TOKEN, output_shape=None, max_seq_len=None)
    )

    assert contract.pooling is PoolingType.LAST_TOKEN
    assert contract.output_shape is None
    assert contract.max_seq_len is None
    manifest = contract.to_manifest_dict()
    assert manifest["output_shape"] is None
    assert manifest["pooling"] == "LAST_TOKEN"
    assert manifest["preprocessor"] == PreprocessorSpec().to_dict()


def test_contract_accepts_multi_axis_shape_with_dynamic_axis_and_no_dim():
    contract = ModelContract(**_contract_kwargs(output_dim=None, output_shape=(None, 128)))

    assert contract.output_shape == (None, 128)
    with pytest.raises(ValueError, match="dense adapter requires output_dim"):
        contract.to_custom_model_spec_kwargs()


@pytest.mark.parametrize(("task", "modality"), [("sparse", "text"), ("dense", "image")])
def test_custom_model_adapter_rejects_non_dense_text_contracts(task: str, modality: str):
    contract = ModelContract(**_contract_kwargs(task=task, modality=modality))

    with pytest.raises(ValueError, match="needs task='dense' and modality='text'"):
        contract.to_custom_model_spec_kwargs()


def test_spec_aliases_point_at_model_contract():
    from fastretrieval.contract import ModelSpec, RerankerSpec

    assert ModelSpec is ModelContract
    assert RerankerSpec is ModelContract


def test_contract_package_exposes_custom_specs_lazily():
    import fastretrieval.contract as contract_package
    from fastretrieval.common.custom_model import CustomModelSpec, CustomRerankerSpec

    assert contract_package.CustomModelSpec is CustomModelSpec
    assert contract_package.CustomRerankerSpec is CustomRerankerSpec
    with pytest.raises(AttributeError, match="'fastretrieval.contract' has no attribute 'Nope'"):
        _ = contract_package.Nope


def test_write_manifest_rejects_metadata_that_overrides_contract_fields(tmp_path: Path):
    contract = ModelContract(**_contract_kwargs())
    path = tmp_path / "manifest.json"

    with pytest.raises(ValueError, match="metadata key conflicts with manifest field: model_id"):
        write_manifest(path, contract, {"model_id": "acme/other"})
    assert not path.exists()


def test_write_manifest_creates_parent_dirs_and_roundtrips(tmp_path: Path):
    contract = ModelContract(**_contract_kwargs())
    path = tmp_path / "nested" / "dir" / "manifest.json"

    write_manifest(path, contract)

    assert load_manifest(path) == contract
    assert [entry.name for entry in path.parent.iterdir()] == ["manifest.json"]
    assert path.read_text(encoding="utf-8").endswith("}\n")


def test_write_manifest_removes_temporary_file_when_serialization_fails(tmp_path: Path):
    contract = ModelContract(**_contract_kwargs())
    path = tmp_path / "manifest.json"

    with pytest.raises(TypeError, match="not JSON serializable"):
        write_manifest(path, contract, {"license": object()})
    assert list(tmp_path.iterdir()) == []


def test_write_manifest_propagates_temp_file_creation_failure_without_leftovers(tmp_path: Path):
    contract = ModelContract(**_contract_kwargs())
    # A file name past the 255-character component limit of every supported filesystem
    # makes the temporary file itself impossible to create.
    path = tmp_path / ("m" * 300 + ".json")

    with pytest.raises(OSError):
        write_manifest(path, contract)
    assert list(tmp_path.iterdir()) == []


def test_load_manifest_reports_unreadable_file(tmp_path: Path):
    with pytest.raises(ValueError, match="cannot read manifest .*missing.json"):
        load_manifest(tmp_path / "missing.json")


def test_load_manifest_reports_malformed_json(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text("{broken", encoding="utf-8")

    with pytest.raises(ValueError, match="cannot read manifest"):
        load_manifest(path)


def test_load_manifest_rejects_non_object_payload(tmp_path: Path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps([_payload()]), encoding="utf-8")

    with pytest.raises(ValueError, match="must be a JSON object"):
        load_manifest(path)


def test_load_manifest_lists_every_missing_field(tmp_path: Path):
    payload = _payload("acme/partial")
    del payload["exporter_version"]
    del payload["quantization"]

    with pytest.raises(
        ValueError, match="acme/partial: manifest is missing exporter_version, quantization"
    ):
        load_manifest(_write_manifest(tmp_path, payload))


def test_load_manifest_uses_placeholder_id_when_model_id_missing(tmp_path: Path):
    payload = _payload()
    del payload["model_id"]

    with pytest.raises(ValueError, match="<unknown>: manifest is missing model_id"):
        load_manifest(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("1", "schema_version must be an integer"),
        (True, "schema_version must be an integer"),
        (2, "unsupported manifest schema_version 2"),
    ],
)
def test_load_manifest_validates_schema_version(tmp_path: Path, value: object, message: str):
    payload = _payload("acme/schema")
    payload["schema_version"] = value

    with pytest.raises(ValueError, match=f"acme/schema: {message}"):
        load_manifest(_write_manifest(tmp_path, payload))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("source", 5, "manifest field source must be a string"),
        ("pooling", None, "manifest field pooling must be a string"),
        ("output_dim", "384", "manifest field output_dim must be an integer or null"),
        ("max_seq_len", True, "manifest field max_seq_len must be an integer or null"),
        ("output_shape", "384", "manifest field output_shape must be a list or null"),
        ("output_shape", ["384"], "manifest output_shape values must be integers or null"),
        ("output_shape", [False], "manifest output_shape values must be integers or null"),
        ("normalization", "true", "manifest field normalization must be boolean"),
        (
            "tokenizer_files",
            "tokenizer.json",
            "manifest field tokenizer_files must be a list of strings",
        ),
        ("artifact_formats", [""], "manifest field artifact_formats must be a list of strings"),
        ("artifact_formats", [1], "manifest field artifact_formats must be a list of strings"),
        ("quantization", 8, "manifest field quantization must be a string or null"),
        ("exporter_version", 1.0, "manifest field exporter_version must be a string or null"),
    ],
)
def test_load_manifest_rejects_wrong_json_types(
    tmp_path: Path, field: str, value: object, message: str
):
    payload = _payload("acme/types")
    payload[field] = value

    with pytest.raises(ValueError, match=f"acme/types: {message}"):
        load_manifest(_write_manifest(tmp_path, payload))


def test_load_manifest_wraps_preprocessor_errors_as_invalid_fields(tmp_path: Path):
    payload = _payload("acme/preprocessor")
    payload["preprocessor"] = "text"

    with pytest.raises(
        ValueError,
        match="acme/preprocessor: invalid manifest fields: preprocessor must be an object",
    ):
        load_manifest(_write_manifest(tmp_path, payload))


def test_load_manifest_accepts_null_dimension_with_dynamic_shape(tmp_path: Path):
    payload = _payload("acme/dynamic")
    payload["output_dim"] = None
    payload["output_shape"] = [None, 128]
    payload["max_seq_len"] = None

    contract = load_manifest(_write_manifest(tmp_path, payload))

    assert contract.output_dim is None
    assert contract.output_shape == (None, 128)
    assert contract.max_seq_len is None
