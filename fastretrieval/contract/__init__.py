"""Hợp đồng khai báo model: người dùng mô tả model, thư viện không giữ zoo."""

from fastretrieval.contract.manifest import load_manifest, write_manifest
from fastretrieval.contract.model import (
    ArtifactFormat,
    ModelContract,
    ModelModality,
    ModelSpec,
    ModelTask,
    RerankerSpec,
)
from fastretrieval.contract.preprocessor import PreprocessorKind, PreprocessorSpec

__all__ = [
    "ArtifactFormat",
    "ModelContract",
    "ModelModality",
    "ModelSpec",
    "ModelTask",
    "PreprocessorKind",
    "PreprocessorSpec",
    "RerankerSpec",
    "load_manifest",
    "write_manifest",
    "CustomModelSpec",
    "CustomRerankerSpec",
]


def __getattr__(name: str):
    """Giữ package contract import-safe cạnh common.custom_model."""
    if name in {"CustomModelSpec", "CustomRerankerSpec"}:
        from fastretrieval.common.custom_model import CustomModelSpec, CustomRerankerSpec

        return {"CustomModelSpec": CustomModelSpec, "CustomRerankerSpec": CustomRerankerSpec}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
