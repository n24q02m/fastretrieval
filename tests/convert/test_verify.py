import numpy as np
import pytest

from fastretrieval.contract import ModelContract, PreprocessorSpec
from fastretrieval.convert.manifest import write_manifest
from fastretrieval.convert.verify import compare_embeddings, verify_converted, verify_manifest


def _write_artifact(
    tmp_path,
    *,
    formats=("onnx",),
    names=("onnx/model.onnx",),
    quantization=None,
    task="dense",
    output_dim=3,
    output_shape=(3,),
    pooling="MEAN",
    normalization=False,
):
    contract = ModelContract(
        model_id="acme/tiny-model",
        source="acme/tiny-model",
        task=task,
        modality="text",
        model_family="bert",
        output_dim=output_dim,
        output_shape=output_shape,
        pooling=pooling,
        normalization=normalization,
        max_seq_len=32,
        preprocessor=PreprocessorSpec(kind="text"),
        artifact_formats=formats,
        tokenizer_files=("config.json", "tokenizer.json", "tokenizer_config.json"),
        quantization=quantization,
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


# ---------------------------------------------------------------------------
# Per-variant tolerance (F2)
# ---------------------------------------------------------------------------
def _stub_verify(monkeypatch, reference, candidate_by_name):
    monkeypatch.setattr("fastretrieval.convert.verify.require_convert_deps", lambda *modules: None)
    monkeypatch.setattr(
        "fastretrieval.convert.verify.validate_artifacts",
        lambda directory, expected_source=None: verify_manifest(
            directory, expected_source=expected_source
        ),
    )
    monkeypatch.setattr(
        "fastretrieval.convert.verify._reference_embeddings",
        lambda source, contract: reference,
    )

    def _fake_onnx(artifact, contract, source):
        return candidate_by_name[artifact.name]

    monkeypatch.setattr("fastretrieval.convert.verify._onnx_embeddings", _fake_onnx)


def test_default_atol_stays_strict_for_full_precision_artifact(tmp_path, monkeypatch):
    _write_artifact(tmp_path, names=("onnx/model.onnx",))
    reference = np.zeros((1, 3), dtype=np.float32)
    # Drift 0.05: below every quantized default, far above the fp32 gate.
    _stub_verify(monkeypatch, reference, {"model.onnx": np.full((1, 3), 0.05, dtype=np.float32)})

    report = verify_converted(tmp_path, "acme/tiny-model")

    assert report["passed"] is False
    assert report["atol_mode"] == "per_variant"
    assert report["variant_reports"]["onnx/model.onnx"]["atol"] == pytest.approx(1e-2)


def test_default_atol_accepts_converter_quantized_variants(tmp_path, monkeypatch):
    _write_artifact(
        tmp_path,
        names=("onnx/model_quantized.onnx", "onnx/model_q4f16.onnx"),
        quantization="int8,q4f16",
    )
    reference = np.zeros((1, 3), dtype=np.float32)
    # Drifts mirror the measured e2e on all-MiniLM-L6-v2 (0.0756 / 0.0764):
    # the old flat 1e-2 gate rejected the converter's own artifacts.
    _stub_verify(
        monkeypatch,
        reference,
        {
            "model_quantized.onnx": np.full((1, 3), 0.075, dtype=np.float32),
            "model_q4f16.onnx": np.full((1, 3), 0.076, dtype=np.float32),
        },
    )

    report = verify_converted(tmp_path, "acme/tiny-model")

    assert report["passed"] is True
    assert report["atol_mode"] == "per_variant"
    variants = report["variant_reports"]
    assert variants["onnx/model_quantized.onnx"]["atol"] == pytest.approx(0.1)
    assert variants["onnx/model_q4f16.onnx"]["atol"] == pytest.approx(0.15)


def test_explicit_atol_override_wins_over_per_variant_defaults(tmp_path, monkeypatch):
    _write_artifact(
        tmp_path,
        names=("onnx/model_quantized.onnx", "onnx/model_q4f16.onnx"),
        quantization="int8,q4f16",
    )
    reference = np.zeros((1, 3), dtype=np.float32)
    candidates = {
        "model_quantized.onnx": np.full((1, 3), 0.075, dtype=np.float32),
        "model_q4f16.onnx": np.full((1, 3), 0.076, dtype=np.float32),
    }
    _stub_verify(monkeypatch, reference, candidates)

    strict = verify_converted(tmp_path, "acme/tiny-model", atol=1e-2)

    assert strict["passed"] is False
    assert strict["atol_mode"] == "override"
    assert all(item["atol"] == pytest.approx(1e-2) for item in strict["variant_reports"].values())

    loose = verify_converted(tmp_path, "acme/tiny-model", atol=0.5)

    assert loose["passed"] is True
    assert loose["atol_mode"] == "override"
    assert loose["atol"] == pytest.approx(0.5)


def test_negative_atol_is_rejected(tmp_path):
    _write_artifact(tmp_path, names=("onnx/model.onnx",))
    with pytest.raises(ValueError, match="non-negative"):
        verify_converted(tmp_path, "acme/tiny-model", atol=-1e-2)


def test_verify_accepts_cross_encoder_manifest(tmp_path, monkeypatch):
    _write_artifact(
        tmp_path,
        task="cross_encoder",
        output_dim=2,
        output_shape=(2,),
        pooling="CLS",
        names=("onnx/model.onnx",),
    )
    reference = np.zeros((1, 2), dtype=np.float32)
    _stub_verify(monkeypatch, reference, {"model.onnx": reference})

    report = verify_converted(tmp_path, "acme/tiny-model")

    assert report["passed"] is True
    assert len(report["variant_reports"]) == 1


def test_verify_still_rejects_tasks_outside_the_supported_set(tmp_path, monkeypatch):
    _write_artifact(
        tmp_path,
        task="generative_reranker",
        output_dim=2,
        output_shape=(2,),
        pooling="CLS",
        names=("onnx/model.onnx",),
    )
    _stub_verify(
        monkeypatch,
        np.zeros((1, 2), dtype=np.float32),
        {"model.onnx": np.zeros((1, 2), dtype=np.float32)},
    )

    with pytest.raises(ValueError, match="supports only task"):
        verify_converted(tmp_path, "acme/tiny-model")
