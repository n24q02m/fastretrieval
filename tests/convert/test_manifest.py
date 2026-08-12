import json
from pathlib import Path

import pytest

from fastretrieval.contract.preprocessor import PreprocessorSpec
from fastretrieval.convert.manifest import load_manifest, write_manifest
from fastretrieval.convert.profiles import resolve_profile

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.mark.parametrize(
    ("fixture", "model_family", "output_dim"),
    [
        ("qwen3", "qwen3", 1024),
        ("tiny-e5", "e5", 384),
    ],
)
def test_reference_and_non_qwen_profiles_build_the_same_contract_shape(
    fixture: str, model_family: str, output_dim: int
):
    source = FIXTURES / fixture
    profile = resolve_profile(source, task="dense", modality="text")

    contract = profile.build_contract(
        pooling="mean",
        normalization=True,
        output_dim=output_dim,
        artifact_formats=("onnx",),
        exporter_version="fixture",
    )

    assert contract.model_family == model_family
    assert contract.task == "dense"
    assert contract.modality == "text"
    assert contract.output_dim == output_dim
    assert contract.output_shape == (output_dim,)
    assert contract.preprocessor.kind == "text"
    assert contract.tokenizer_files == (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
    )


def test_manifest_round_trip_is_atomic_and_does_not_invent_provenance(tmp_path: Path):
    source = FIXTURES / "tiny-e5"
    contract = resolve_profile(source, task="dense", modality="text").build_contract(
        pooling="mean",
        normalization=True,
        output_dim=384,
        artifact_formats=("onnx",),
        exporter_version="fixture",
    )

    manifest = write_manifest(tmp_path, contract, metadata={"fixture": True})

    assert manifest == tmp_path / "fastretrieval-manifest.json"
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["fixture"] is True
    assert payload["source"] == str(source)
    assert payload["model_family"] == "e5"
    assert "license" not in payload
    assert "provenance" not in payload
    assert list(payload) == sorted(payload)
    assert load_manifest(tmp_path) == contract


def test_manifest_rejects_missing_required_fields(tmp_path: Path):
    manifest = tmp_path / "fastretrieval-manifest.json"
    manifest.write_text(json.dumps({"schema_version": 1}), encoding="utf-8")

    with pytest.raises(ValueError, match="manifest is missing"):
        load_manifest(tmp_path)


def test_unknown_architecture_fails_closed_with_context(tmp_path: Path):
    source = tmp_path / "unknown"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "unknown_architecture", "hidden_size": 32}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"source=.*task=.*modality="):
        resolve_profile(source, task="dense", modality="text")


def test_unsupported_modality_fails_closed_with_context():
    source = FIXTURES / "tiny-e5"

    with pytest.raises(ValueError, match=r"source=.*task=.*modality=image"):
        resolve_profile(source, task="dense", modality="image")


def test_missing_pooling_or_normalization_fails_closed_with_context():
    source = FIXTURES / "tiny-e5"
    profile = resolve_profile(source, task="dense", modality="text")

    with pytest.raises(ValueError, match=r"source=.*task=.*modality=.*pooling"):
        profile.build_contract(
            pooling=None,
            normalization=True,
            output_dim=384,
            artifact_formats=("onnx",),
        )

    with pytest.raises(ValueError, match=r"source=.*task=.*modality=.*normalization"):
        profile.build_contract(
            pooling="mean",
            normalization=None,
            output_dim=384,
            artifact_formats=("onnx",),
        )


def test_contract_rejects_unsupported_format_and_mismatched_preprocessor():
    source = FIXTURES / "tiny-e5"
    profile = resolve_profile(source, task="dense", modality="text")

    with pytest.raises(ValueError, match="outside the profile"):
        profile.build_contract(
            pooling="mean",
            normalization=True,
            output_dim=384,
            artifact_formats=("gguf",),
        )

    with pytest.raises(ValueError, match="preprocessor kind 'image'.*modality 'text'"):
        profile.build_contract(
            pooling="mean",
            normalization=True,
            output_dim=384,
            artifact_formats=("onnx",),
            preprocessor=PreprocessorSpec(kind="image"),
        )

    with pytest.raises(ValueError, match="artifact_formats must not contain duplicates"):
        profile.build_contract(
            pooling="mean",
            normalization=True,
            output_dim=384,
            artifact_formats=("onnx", "onnx"),
        )

    with pytest.raises(ValueError, match="tokenizer_files must not contain duplicates"):
        profile.build_contract(
            pooling="mean",
            normalization=True,
            output_dim=384,
            artifact_formats=("onnx",),
            tokenizer_files=("tokenizer.json", "tokenizer.json"),
        )


def test_ambiguous_architecture_fails_closed(tmp_path: Path):
    source = tmp_path / "ambiguous"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bert",
                "architectures": ["BertModel", "Qwen3ForCausalLM"],
                "hidden_size": 32,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="matches multiple support profiles"):
        resolve_profile(source, task="dense", modality="text")


def test_invalid_input_names_fail_closed(tmp_path: Path):
    source = tmp_path / "invalid-input-names"
    source.mkdir()
    (source / "config.json").write_text(
        json.dumps({"model_type": "bert", "input_names": "input_ids", "hidden_size": 32}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="input_names must be a sequence"):
        resolve_profile(source, task="dense", modality="text")


def test_cross_encoder_requires_explicit_output_contract():
    profile = resolve_profile(FIXTURES / "tiny-e5", task="cross_encoder", modality="text")

    with pytest.raises(ValueError, match="requires explicit output_dim/output_shape"):
        profile.build_contract(
            pooling="mean",
            normalization=False,
            artifact_formats=("onnx",),
        )


def test_manifest_rejects_boolean_schema_version_and_null_model_id(tmp_path: Path):
    source = FIXTURES / "tiny-e5"
    contract = resolve_profile(source, task="dense", modality="text").build_contract(
        pooling="mean",
        normalization=True,
        output_dim=384,
        artifact_formats=("onnx",),
        exporter_version="fixture",
    )
    base = {"schema_version": 1, **contract.to_manifest_dict()}

    boolean_schema = tmp_path / "boolean-schema"
    boolean_schema.mkdir()
    (boolean_schema / "fastretrieval-manifest.json").write_text(
        json.dumps({**base, "schema_version": True}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="schema_version must be an integer"):
        load_manifest(boolean_schema)

    null_model_id = tmp_path / "null-model-id"
    null_model_id.mkdir()
    (null_model_id / "fastretrieval-manifest.json").write_text(
        json.dumps({**base, "model_id": None}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="model_id must be a string"):
        load_manifest(null_model_id)


def test_manifest_writer_cleans_temporary_file_after_serialization_failure(tmp_path: Path):
    source = FIXTURES / "tiny-e5"
    contract = resolve_profile(source, task="dense", modality="text").build_contract(
        pooling="mean",
        normalization=True,
        output_dim=384,
        artifact_formats=("onnx",),
    )

    with pytest.raises(TypeError, match="not JSON serializable"):
        write_manifest(tmp_path, contract, metadata={"fixture": object()})

    assert not list(tmp_path.glob(".fastretrieval-manifest.json.*.tmp"))
