import numpy as np
import pytest

from fastretrieval.contract import ModelContract, PreprocessorSpec
from fastretrieval.convert.manifest import write_manifest
from fastretrieval.convert.verify import compare_embeddings, verify_converted, verify_manifest


def _write_artifact(tmp_path, *, formats=("onnx",), names=("onnx/model.onnx",)):
    contract = ModelContract(
        model_id="acme/tiny-model",
        source="acme/tiny-model",
        task="dense",
        modality="text",
        model_family="bert",
        output_dim=3,
        output_shape=(3,),
        pooling="MEAN",
        normalization=False,
        max_seq_len=32,
        preprocessor=PreprocessorSpec(kind="text"),
        artifact_formats=formats,
        tokenizer_files=("config.json", "tokenizer.json", "tokenizer_config.json"),
    )
    for relative in contract.tokenizer_files:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}", encoding="utf-8")
    for relative in names:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"artifact")
    write_manifest(tmp_path, contract)
    return contract


def test_identical_embeddings_pass():
    a = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    report = compare_embeddings(a, a.copy(), atol=1e-4)
    assert report["passed"] is True
    assert report["max_abs_diff"] == pytest.approx(0.0)


def test_small_drift_still_passes_within_tolerance():
    a = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    b = a + 5e-3
    report = compare_embeddings(a, b, atol=1e-2)
    assert report["passed"] is True
    assert report["cosine"] == pytest.approx(1.0, abs=1e-3)


def test_wrong_normalization_is_caught():
    a = np.array([[0.1, 0.2, 0.3]], dtype=np.float32)
    b = a * 3.0 + 0.5
    report = compare_embeddings(a, b, atol=1e-2)
    assert report["passed"] is False
    assert report["max_abs_diff"] > 1e-2


def test_shape_mismatch_is_an_error_not_a_score():
    a = np.zeros((1, 3), dtype=np.float32)
    b = np.zeros((1, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        compare_embeddings(a, b, atol=1e-2)


def test_manifest_discovers_every_onnx_file_not_only_canonical_names(tmp_path):
    _write_artifact(
        tmp_path,
        names=("onnx/model_quantized.onnx", "variants/custom-q4.onnx"),
    )

    contract = verify_manifest(tmp_path, expected_source="acme/tiny-model")

    assert contract.artifact_formats == ("onnx",)


def test_manifest_rejects_undeclared_artifact_format(tmp_path):
    _write_artifact(tmp_path, formats=("onnx",), names=("model.gguf",))

    with pytest.raises(ValueError, match="does not declare existing format"):
        verify_manifest(tmp_path)


def test_verify_checks_every_onnx_variant(tmp_path, monkeypatch):
    _write_artifact(
        tmp_path,
        names=("onnx/model_quantized.onnx", "onnx/model_q4f16.onnx"),
    )
    monkeypatch.setattr("fastretrieval.convert.verify.require_convert_deps", lambda *modules: None)
    monkeypatch.setattr(
        "fastretrieval.convert.verify.validate_artifacts",
        lambda directory, expected_source=None: verify_manifest(
            directory, expected_source=expected_source
        ),
    )
    monkeypatch.setattr(
        "fastretrieval.convert.verify._reference_embeddings",
        lambda source, contract: np.zeros((1, 3), dtype=np.float32),
    )
    monkeypatch.setattr(
        "fastretrieval.convert.verify._onnx_embeddings",
        lambda artifact, contract, source: np.zeros((1, 3), dtype=np.float32),
    )

    report = verify_converted(tmp_path, "acme/tiny-model")

    assert report["passed"] is True
    assert len(report["variant_reports"]) == 2
