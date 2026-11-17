"""verify.py against real ONNX graphs, real tokenizers and real manifests.

The torch reference and the ONNX Runtime session are the only stand-ins; both come from
``test_conversion_pipelines``, where each installer names its boundary.
"""

import shutil
import sys
from types import SimpleNamespace
from typing import cast

import numpy as np
import pytest
from onnx import TensorProto, helper

from fastretrieval.common.model_description import PoolingType
from fastretrieval.contract import ModelContract, ModelTask, PreprocessorSpec
from fastretrieval.convert.cli import main
from fastretrieval.convert.manifest import write_manifest
from fastretrieval.convert.verify import (
    DEFAULT_FP32_ATOL,
    DEFAULT_VARIANT_ATOL,
    PROBES,
    _normalize,
    _onnx_artifacts,
    _onnx_embeddings,
    _pool_hidden,
    _pool_numpy,
    _validate_output_shape,
    compare_embeddings,
    validate_artifacts,
    verify_converted,
    verify_manifest,
)
from tests.convert.test_conversion_pipelines import (
    CLASSIFIER,
    EMBEDDINGS,
    HIDDEN,
    TOKENIZER_FILES,
    FakeTensor,
    hidden_state_graph,
    install_fake_onnxruntime,
    install_fake_torch,
    install_fake_transformers,
    save_model,
    token_inputs,
    token_logits_graph,
    write_model_dir,
)

POOL_IDS = np.array([[2, 3, 6], [7, 4, 5]], dtype=np.int64)
# The second row is fully masked: pooling must clamp instead of dividing by zero.
POOL_MASK = np.array([[1, 1, 0], [0, 0, 0]], dtype=np.int64)
POOL_HIDDEN = EMBEDDINGS[POOL_IDS]
# _pool_numpy types the mask as an array; None proves these paths never read it.
NO_MASK = cast(np.ndarray, None)


@pytest.fixture
def runtime(monkeypatch):
    """The verify boundaries; each installer documents its own boundary."""
    return SimpleNamespace(
        ort=install_fake_onnxruntime(monkeypatch),
        torch=install_fake_torch(monkeypatch),
        transformers=install_fake_transformers(monkeypatch),
    )


@pytest.fixture
def model_dir(tmp_path):
    return write_model_dir(tmp_path / "tiny-bert", model_type="bert")


def make_contract(
    source,
    *,
    task: ModelTask = "dense",
    output_dim: int | None = HIDDEN,
    output_shape: tuple[int | None, ...] | None = (HIDDEN,),
    pooling: PoolingType | str = "MEAN",
    normalization: bool = False,
    max_seq_len: int | None = 16,
    artifact_formats: tuple[str, ...] = ("onnx",),
):
    return ModelContract(
        model_id=str(source),
        source=str(source),
        task=task,
        modality="text",
        model_family="bert",
        output_dim=output_dim,
        output_shape=output_shape,
        pooling=pooling,
        normalization=normalization,
        max_seq_len=max_seq_len,
        preprocessor=PreprocessorSpec(kind="text"),
        artifact_formats=artifact_formats,
        tokenizer_files=TOKENIZER_FILES,
    )


def write_converted(directory, contract, model_dir, artifacts):
    """Lay out a converted directory: tokenizer files, artifacts, then the manifest."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in contract.tokenizer_files:
        shutil.copy2(model_dir / name, directory / name)
    for relative, content in artifacts.items():
        path = directory / relative
        if isinstance(content, bytes):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        else:
            save_model(content, path)
    write_manifest(directory, contract)
    return directory


def no_output_graph():
    node = helper.make_node("Identity", ["input_ids"], ["unused"])
    return helper.make_graph([node], "no_outputs", token_inputs(), [])


# ---------------------------------------------------------------------------
# verify_manifest and validate_artifacts
# ---------------------------------------------------------------------------
def test_verify_manifest_rejects_a_missing_directory(tmp_path):
    with pytest.raises(ValueError, match="converted artifact directory does not exist"):
        verify_manifest(tmp_path / "absent")


def test_verify_manifest_reports_missing_tokenizer_files(tmp_path, model_dir):
    converted = write_converted(
        tmp_path / "converted",
        make_contract(model_dir),
        model_dir,
        {"onnx/model.onnx": hidden_state_graph()},
    )
    (converted / "tokenizer_config.json").unlink()

    with pytest.raises(ValueError, match="manifest tokenizer files are missing: tokenizer_config"):
        verify_manifest(converted)


def test_verify_manifest_rejects_a_declared_format_without_artifacts(tmp_path, model_dir):
    converted = write_converted(tmp_path / "converted", make_contract(model_dir), model_dir, {})

    with pytest.raises(ValueError, match="declares artifact format\\(s\\) with no artifact: onnx"):
        verify_manifest(converted)


def test_validate_artifacts_wraps_an_unreadable_onnx_file(tmp_path, model_dir, runtime):
    converted = write_converted(
        tmp_path / "converted",
        make_contract(model_dir),
        model_dir,
        {"onnx/model.onnx": b"not a protobuf"},
    )

    with pytest.raises(ValueError, match="invalid ONNX artifact .*model\\.onnx") as caught:
        validate_artifacts(converted)

    assert caught.value.__cause__ is not None
    assert runtime.ort.sessions == []


def test_validate_artifacts_rejects_a_graph_without_inputs(tmp_path, model_dir, runtime):
    constant = helper.make_node(
        "Constant",
        [],
        ["values"],
        value=helper.make_tensor("ones", TensorProto.FLOAT, [HIDDEN], [1.0] * HIDDEN),
    )
    graph = helper.make_graph(
        [constant],
        "no_inputs",
        [],
        [helper.make_tensor_value_info("values", TensorProto.FLOAT, [HIDDEN])],
    )
    converted = write_converted(
        tmp_path / "converted", make_contract(model_dir), model_dir, {"onnx/model.onnx": graph}
    )

    with pytest.raises(ValueError, match="ONNX artifact has no inputs"):
        validate_artifacts(converted)


def test_validate_artifacts_rejects_a_graph_without_outputs(tmp_path, model_dir, runtime):
    converted = write_converted(
        tmp_path / "converted",
        make_contract(model_dir),
        model_dir,
        {"onnx/model.onnx": no_output_graph()},
    )

    with pytest.raises(ValueError, match="ONNX artifact has no outputs"):
        validate_artifacts(converted)


def test_validate_artifacts_skips_onnx_runtime_for_a_gguf_only_artifact(
    tmp_path, model_dir, monkeypatch
):
    # Boundary (a): a None entry proves the GGUF path never needs onnxruntime.
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    contract = make_contract(model_dir, artifact_formats=("gguf",))
    converted = write_converted(
        tmp_path / "converted", contract, model_dir, {"tiny-q4-k-m.gguf": b"GGUF"}
    )

    assert validate_artifacts(converted, expected_source=str(model_dir)) == contract
    with pytest.raises(ValueError, match="ONNX artifacts are not declared"):
        verify_converted(converted, str(model_dir))


def test_onnx_artifacts_requires_at_least_one_onnx_file(tmp_path, model_dir):
    with pytest.raises(FileNotFoundError, match="no ONNX artifact found under"):
        _onnx_artifacts(tmp_path, make_contract(model_dir))


# ---------------------------------------------------------------------------
# Pooling, normalization and output shape
# ---------------------------------------------------------------------------
EXPECTED_POOLING = {
    "CLS": POOL_HIDDEN[:, 0, :],
    "MEAN": np.stack([POOL_HIDDEN[0, :2].mean(axis=0), np.zeros(HIDDEN, np.float32)]),
    "LAST_TOKEN": np.stack([POOL_HIDDEN[0, 1], POOL_HIDDEN[1, 0]]),
}


@pytest.mark.parametrize(
    "pooling",
    [PoolingType.CLS, "cls", PoolingType.MEAN, "mean", PoolingType.LAST_TOKEN, "last_token"],
)
def test_torch_and_numpy_pooling_agree_on_every_strategy(pooling, runtime):
    expected = EXPECTED_POOLING[getattr(pooling, "value", pooling).upper()]

    pooled_torch = _pool_hidden(FakeTensor(POOL_HIDDEN), FakeTensor(POOL_MASK), pooling)
    pooled_numpy = _pool_numpy(POOL_HIDDEN, POOL_MASK, pooling)

    np.testing.assert_allclose(np.asarray(pooled_torch), expected, rtol=1e-6)
    np.testing.assert_allclose(pooled_numpy, expected, rtol=1e-6)


def test_pooling_returns_two_dimensional_outputs_unchanged(runtime):
    logits = FakeTensor(np.ones((2, 1), np.float32))
    array = np.ones((2, 1), np.float32)

    assert _pool_hidden(logits, None, "mean") is logits
    assert _pool_numpy(array, NO_MASK, "mean") is array


def test_pooling_rejects_outputs_that_are_not_two_or_three_dimensional(runtime):
    values = np.zeros((1, 1, 1, 1), np.float32)

    with pytest.raises(ValueError, match=r"2D or 3D transformer output, got shape \(1, 1, 1, 1\)"):
        _pool_hidden(FakeTensor(values), None, "mean")
    with pytest.raises(ValueError, match=r"2D or 3D ONNX output, got shape \(1, 1, 1, 1\)"):
        _pool_numpy(values, NO_MASK, "mean")


@pytest.mark.parametrize(
    ("pooling", "message"),
    [
        (PoolingType.DISABLED, "pooling=DISABLED cannot produce dense vectors"),
        ("max", "unsupported pooling 'max'"),
    ],
)
def test_pooling_rejects_strategies_that_cannot_build_a_dense_vector(pooling, message, runtime):
    with pytest.raises(ValueError, match=message):
        _pool_hidden(FakeTensor(POOL_HIDDEN), FakeTensor(POOL_MASK), pooling)
    with pytest.raises(ValueError, match=message):
        _pool_numpy(POOL_HIDDEN, POOL_MASK, pooling)


def test_normalize_disabled_returns_the_input_object():
    values = np.array([[3.0, 4.0]])

    assert _normalize(values, False) is values


def test_normalize_hands_torch_tensors_to_torch_l2_normalize(runtime):
    normalized = _normalize(FakeTensor(np.array([[3.0, 4.0]], np.float32)), True)

    assert runtime.torch.normalize == [{"p": 2, "dim": -1}]
    np.testing.assert_allclose(np.asarray(normalized), [[0.6, 0.8]])


def test_normalize_scales_numpy_rows_to_unit_length_and_keeps_zero_rows(runtime):
    normalized = _normalize(np.array([[3.0, 4.0], [0.0, 0.0]]), True)

    assert normalized.dtype == np.float32
    np.testing.assert_allclose(normalized, [[0.6, 0.8], [0.0, 0.0]])
    assert runtime.torch.normalize == []


@pytest.mark.parametrize(
    ("vector", "expected"), [([3.0, 4.0], [0.6, 0.8]), ([0.0, 0.0], [0.0, 0.0])]
)
def test_normalize_handles_a_single_vector(vector, expected, runtime):
    np.testing.assert_allclose(_normalize(np.array(vector), True), expected)


@pytest.mark.parametrize(
    ("overrides", "shape", "message"),
    [
        ({}, (2, 1, HIDDEN), r"reference output must be 2D, got \(2, 1, 4\)"),
        (
            {"output_dim": None, "output_shape": (2, 2)},
            (2, HIDDEN),
            r"reference output shape mismatch: manifest \(2, 2\), got \(4,\)",
        ),
        (
            {"output_dim": None},
            (2, 3),
            r"reference output shape mismatch: manifest \(4,\), got \(3,\)",
        ),
        (
            {"output_shape": None},
            (2, 3),
            "reference output dimension mismatch: manifest 4, got 3",
        ),
    ],
)
def test_output_shape_must_match_the_manifest(overrides, shape, message, model_dir):
    contract = make_contract(model_dir, **overrides)

    with pytest.raises(ValueError, match=message):
        _validate_output_shape(np.zeros(shape, np.float32), contract, "reference")


@pytest.mark.parametrize(
    "overrides",
    [{"output_dim": None, "output_shape": (None,)}, {"output_shape": None}],
)
def test_output_shape_accepts_open_dimensions_and_dim_only_manifests(overrides, model_dir):
    contract = make_contract(model_dir, **overrides)
    width = 7 if contract.output_dim is None else HIDDEN

    assert _validate_output_shape(np.zeros((2, width), np.float32), contract, "x") is None


def test_compare_embeddings_rejects_a_negative_tolerance():
    with pytest.raises(ValueError, match="atol must be non-negative"):
        compare_embeddings([1.0], [1.0], atol=-0.1)


@pytest.mark.parametrize("values", [np.zeros((2, 3)), np.zeros((0, 3))])
def test_compare_embeddings_scores_zero_and_empty_arrays_as_identical(values):
    report = compare_embeddings(values, values.copy(), atol=0.0)

    assert report == {
        "passed": True,
        "max_abs_diff": 0.0,
        "mean_abs_diff": 0.0,
        "cosine": 1.0,
        "atol": 0.0,
    }


# ---------------------------------------------------------------------------
# verify_converted against the torch reference
# ---------------------------------------------------------------------------
def test_verify_converted_flags_the_variant_that_drifts_from_the_reference(
    tmp_path, model_dir, runtime
):
    converted = write_converted(
        tmp_path / "converted",
        make_contract(model_dir, normalization=True),
        model_dir,
        {
            "onnx/model.onnx": hidden_state_graph(),
            "onnx/model_quantized.onnx": hidden_state_graph(-EMBEDDINGS),
        },
    )

    report = verify_converted(converted, str(model_dir))

    fp32 = report["variant_reports"]["onnx/model.onnx"]
    int8 = report["variant_reports"]["onnx/model_quantized.onnx"]
    assert report["passed"] is False
    assert (fp32["passed"], fp32["atol"]) == (True, DEFAULT_FP32_ATOL)
    assert fp32["max_abs_diff"] < 1e-6
    assert fp32["cosine"] == pytest.approx(1.0)
    assert (int8["passed"], int8["atol"]) == (False, DEFAULT_VARIANT_ATOL["int8"])
    assert int8["cosine"] == pytest.approx(-1.0)
    assert report["max_abs_diff"] == int8["max_abs_diff"]
    assert report["cosine"] == int8["cosine"]
    assert (report["atol"], report["atol_mode"], report["atol_range"]) == (
        None,
        "per_variant",
        (DEFAULT_FP32_ATOL, DEFAULT_VARIANT_ATOL["int8"]),
    )
    assert report["probes"] == len(PROBES)
    assert report["artifacts"] == [
        str(converted / "onnx" / "model.onnx"),
        str(converted / "onnx" / "model_quantized.onnx"),
    ]
    assert report["artifact"] == report["artifacts"][0]
    assert report["model_id"] == str(model_dir)

    source = str(model_dir)
    assert runtime.transformers.loads == [
        ("AutoTokenizer", source, {}),
        ("AutoModel", source, {"torch_dtype": np.float32}),
        ("AutoTokenizer", source, {}),
        ("AutoTokenizer", source, {}),
    ]
    tokenizer_options = {"padding": True, "truncation": True, "max_length": 16}
    assert runtime.transformers.tokenizer_calls == [
        (PROBES, {"return_tensors": "pt"} | tokenizer_options),
        (PROBES, {"return_tensors": "np"} | tokenizer_options),
        (PROBES, {"return_tensors": "np"} | tokenizer_options),
    ]
    assert runtime.torch.normalize == [{"p": 2, "dim": -1}]
    assert [path.name for path, _ in runtime.ort.sessions] == [
        "model.onnx",
        "model_quantized.onnx",
        "model.onnx",
        "model_quantized.onnx",
    ]


def write_cross_encoder(tmp_path, model_dir, classifier):
    contract = make_contract(
        model_dir,
        task="cross_encoder",
        pooling="CLS",
        output_dim=1,
        output_shape=(1,),
        max_seq_len=None,
    )
    graph = token_logits_graph(EMBEDDINGS, classifier, position=0)
    return write_converted(
        tmp_path / "converted", contract, model_dir, {"onnx/model_quantized.onnx": graph}
    )


def printed_metrics(output):
    lines = [line.split(": ") for line in output.splitlines()]
    return {key: float(value) for key, value in lines}


def test_cli_verify_passes_a_cross_encoder_matching_its_reference_logits(
    tmp_path, model_dir, runtime, capsys
):
    converted = write_cross_encoder(tmp_path, model_dir, CLASSIFIER)

    exit_code = main(["verify", str(converted), "--source", str(model_dir)])

    assert exit_code == 0
    metrics = printed_metrics(capsys.readouterr().out)
    assert list(metrics) == ["max_abs_diff", "mean_abs_diff", "cosine"]
    assert metrics["max_abs_diff"] < 1e-5
    assert metrics["cosine"] == pytest.approx(1.0)
    assert runtime.transformers.loads[1] == (
        "AutoModelForSequenceClassification",
        str(model_dir),
        {"torch_dtype": np.float32},
    )
    assert runtime.transformers.tokenizer_calls[0] == (
        PROBES,
        {"return_tensors": "pt", "padding": True, "truncation": True},
    )
    assert runtime.torch.normalize == []


def test_cli_verify_exits_with_one_when_logits_drift_past_the_tolerance(
    tmp_path, model_dir, runtime, capsys
):
    converted = write_cross_encoder(tmp_path, model_dir, CLASSIFIER + 1.0)

    exit_code = main(["verify", str(converted), "--source", str(model_dir), "--atol", "0.001"])

    assert exit_code == 1
    assert printed_metrics(capsys.readouterr().out)["max_abs_diff"] > 0.001


def test_verify_converted_names_onnx_inputs_the_tokenizer_cannot_fill(
    tmp_path, model_dir, runtime
):
    graph = hidden_state_graph(input_names=("input_ids", "attention_mask", "token_type_ids"))
    converted = write_converted(
        tmp_path / "converted", make_contract(model_dir), model_dir, {"onnx/model.onnx": graph}
    )

    with pytest.raises(
        ValueError, match="tokenizer cannot provide ONNX input\\(s\\): token_type_ids"
    ):
        verify_converted(converted, str(model_dir))


def test_onnx_embeddings_fails_when_the_session_returns_no_outputs(tmp_path, model_dir, runtime):
    artifact = save_model(no_output_graph(), tmp_path / "model.onnx")

    with pytest.raises(ValueError, match="ONNX session returned no outputs"):
        _onnx_embeddings(artifact, make_contract(model_dir), str(model_dir))
