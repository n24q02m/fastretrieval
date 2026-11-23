"""Public API and packaging surface contract tests for fastretrieval v2."""

from __future__ import annotations

import tomllib
from pathlib import Path

import fastretrieval
from fastretrieval.common.model_description import PoolingType
from fastretrieval.contract.model import ModelContract, ModelSpec, RerankerSpec
from fastretrieval.contract.preprocessor import PreprocessorSpec

EXPECTED_PUBLIC_ALL = [
    "CustomModelSpec",
    "CustomRerankerSpec",
    "Device",
    "define_cache_dir",
    "ImageEmbedding",
    "LateInteractionTextEmbedding",
    "LateInteractionMultimodalEmbedding",
    "SparseEmbedding",
    "SparseTextEmbedding",
    "TextEmbedding",
    "TextCrossEncoder",
]

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_public_all_exports_are_complete_and_importable() -> None:
    assert fastretrieval.__all__ == EXPECTED_PUBLIC_ALL
    for symbol in EXPECTED_PUBLIC_ALL:
        assert hasattr(fastretrieval, symbol), f"missing export: {symbol}"


def test_model_contract_aliases_and_serialization() -> None:
    assert ModelSpec is ModelContract
    assert RerankerSpec is ModelContract

    contract = ModelContract(
        model_id="org/dense-text-v2",
        source="org/dense-text-v2",
        task="dense",
        modality="text",
        model_family="qwen3",
        output_dim=1024,
        output_shape=None,
        pooling=PoolingType.MEAN,
        normalization=True,
        max_seq_len=512,
        preprocessor=PreprocessorSpec(kind="text"),
        artifact_formats=("onnx",),
        tokenizer_files=("tokenizer.json",),
    )
    adapter = contract.to_custom_model_spec_kwargs()
    manifest = contract.to_manifest_dict()

    assert adapter["pooling"] == "MEAN"
    assert adapter["dim"] == 1024
    assert manifest["pooling"] == "MEAN"
    assert manifest["preprocessor"] == {
        "config_file": None,
        "image_mean": None,
        "image_size": None,
        "image_std": None,
        "kind": "text",
    }


def test_preprocessor_from_dict_validates_tuple_lengths() -> None:
    import pytest

    with pytest.raises(ValueError, match="image_size must contain two positive values"):
        PreprocessorSpec.from_dict({"kind": "image", "image_size": [224]})
    with pytest.raises(ValueError, match="image_mean must contain three channel values"):
        PreprocessorSpec.from_dict({"kind": "image", "image_mean": [0.5, 0.5]})
    with pytest.raises(ValueError, match="image_std must contain three non-zero values"):
        PreprocessorSpec.from_dict({"kind": "image", "image_std": [0.5]})


def test_pyproject_scripts_and_unbounded_ty_dependency() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = pyproject["project"]["scripts"]
    assert scripts["fastretrieval-server"] == "fastretrieval.server.__main__:main"

    dev_deps = pyproject["dependency-groups"]["dev"]
    ty_specs = [dep for dep in dev_deps if dep.startswith("ty")]
    assert ty_specs, "ty must remain in dev dependency group"
    assert all("<0.0.105" not in dep for dep in ty_specs)
