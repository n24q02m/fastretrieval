import json
from pathlib import Path

import pytest

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
    assert contract.tokenizer_files == ("tokenizer.json",)


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
