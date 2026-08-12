"""Contract capability/task/modality cho artifact retrieval đa model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from fastretrieval.common.model_description import PoolingType
from fastretrieval.contract.preprocessor import PreprocessorSpec

ModelTask = Literal[
    "dense",
    "sparse",
    "late_interaction",
    "cross_encoder",
    "generative_reranker",
]
ModelModality = Literal["text", "image", "audio", "video"]
ArtifactFormat = Literal["onnx", "gguf"]

_MODEL_TASKS = frozenset(
    {"dense", "sparse", "late_interaction", "cross_encoder", "generative_reranker"}
)
_MODEL_MODALITIES = frozenset({"text", "image", "audio", "video"})
_ARTIFACT_FORMATS = frozenset({"onnx", "gguf"})


@dataclass(frozen=True)
class ModelContract:
    """Mô tả đầy đủ capability của một model artifact.

    Những trường quyết định cách chạy đều bắt buộc. Contract không chứa
    registry model cụ thể và không suy luận capability từ tên model.
    """

    model_id: str
    source: str
    task: ModelTask
    modality: ModelModality
    model_family: str
    output_dim: int | None
    output_shape: tuple[int | None, ...] | None
    pooling: PoolingType | str
    normalization: bool
    max_seq_len: int | None
    preprocessor: PreprocessorSpec
    artifact_formats: tuple[ArtifactFormat | str, ...]
    tokenizer_files: tuple[str, ...]
    quantization: str | None = None
    exporter_version: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model_id, str) or not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError(f"{self.model_id}: source must not be empty")
        if not isinstance(self.model_family, str) or not self.model_family.strip():
            raise ValueError(f"{self.model_id}: model_family must not be empty")
        if not isinstance(self.task, str) or self.task not in _MODEL_TASKS:
            raise ValueError(f"{self.model_id}: unsupported task {self.task!r}")
        if not isinstance(self.modality, str) or self.modality not in _MODEL_MODALITIES:
            raise ValueError(f"{self.model_id}: unsupported modality {self.modality!r}")
        try:
            pooling = (
                self.pooling
                if isinstance(self.pooling, PoolingType)
                else PoolingType(str(self.pooling).upper())
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{self.model_id}: unsupported pooling {self.pooling!r}") from exc
        object.__setattr__(self, "pooling", pooling)
        if not isinstance(self.normalization, bool):
            raise ValueError(f"{self.model_id}: normalization must be boolean")
        if self.output_dim is None and self.output_shape is None:
            raise ValueError(f"{self.model_id}: output_dim or output_shape is required")
        if self.output_dim is not None and (
            not isinstance(self.output_dim, int)
            or isinstance(self.output_dim, bool)
            or self.output_dim <= 0
        ):
            raise ValueError(f"{self.model_id}: output_dim must be positive")
        if self.output_shape is not None:
            if not isinstance(self.output_shape, (tuple, list)):
                raise ValueError(f"{self.model_id}: output_shape must be a tuple or list")
            if not self.output_shape:
                raise ValueError(f"{self.model_id}: output_shape must not be empty")
            if any(
                value is not None
                and (not isinstance(value, int) or isinstance(value, bool) or value <= 0)
                for value in self.output_shape
            ):
                raise ValueError(f"{self.model_id}: output_shape values must be positive")
            if (
                self.output_dim is not None
                and len(self.output_shape) == 1
                and self.output_shape[0] is not None
                and self.output_shape[0] != self.output_dim
            ):
                raise ValueError(f"{self.model_id}: output_dim disagrees with output_shape")
        if self.max_seq_len is not None and (
            not isinstance(self.max_seq_len, int)
            or isinstance(self.max_seq_len, bool)
            or self.max_seq_len <= 0
        ):
            raise ValueError(f"{self.model_id}: max_seq_len must be positive")
        if not isinstance(self.preprocessor, PreprocessorSpec):
            raise ValueError(f"{self.model_id}: preprocessor must be a PreprocessorSpec")
        if not isinstance(self.artifact_formats, (tuple, list)):
            raise ValueError(f"{self.model_id}: artifact_formats must be a tuple or list")
        formats = tuple(self.artifact_formats)
        if not formats:
            raise ValueError(f"{self.model_id}: artifact_formats is required")
        unsupported = [item for item in formats if item not in _ARTIFACT_FORMATS]
        if unsupported:
            raise ValueError(f"{self.model_id}: unsupported artifact_formats {unsupported!r}")
        if len(set(formats)) != len(formats):
            raise ValueError(f"{self.model_id}: artifact_formats must not contain duplicates")
        object.__setattr__(self, "artifact_formats", formats)
        if not isinstance(self.tokenizer_files, (tuple, list)):
            raise ValueError(f"{self.model_id}: tokenizer_files must be a tuple or list")
        if any(not isinstance(item, str) or not item for item in self.tokenizer_files):
            raise ValueError(f"{self.model_id}: tokenizer_files must contain non-empty strings")
        if len(set(self.tokenizer_files)) != len(self.tokenizer_files):
            raise ValueError(f"{self.model_id}: tokenizer_files must not contain duplicates")
        if self.output_shape is not None:
            object.__setattr__(self, "output_shape", tuple(self.output_shape))
        object.__setattr__(self, "tokenizer_files", tuple(self.tokenizer_files))

    @classmethod
    def from_manifest(cls, path: Path) -> ModelContract:
        """Đọc manifest qua một code path chung cho Qwen và non-Qwen."""
        from fastretrieval.contract.manifest import load_manifest

        return load_manifest(path)

    def to_custom_model_spec_kwargs(
        self,
        *,
        model_file: str = "onnx/model.onnx",
        hf: str | None = None,
    ) -> dict[str, Any]:
        """Chuyển contract dense text sang adapter runtime hiện hữu."""
        if self.task != "dense" or self.modality != "text":
            raise ValueError(
                f"{self.model_id}: CustomModelSpec adapter needs task='dense' and modality='text'"
            )
        if self.output_dim is None:
            raise ValueError(f"{self.model_id}: dense adapter requires output_dim")
        return {
            "model_id": self.model_id,
            "hf": self.source if hf is None else hf,
            "model_file": model_file,
            "dim": self.output_dim,
            "pooling": self.pooling.value,
            "normalization": self.normalization,
            "max_seq_len": self.max_seq_len,
            "preprocessor": self.preprocessor,
        }

    def to_manifest_dict(self) -> dict[str, Any]:
        """Serialize contract fields without artifact-source metadata."""
        return {
            "artifact_formats": list(self.artifact_formats),
            "exporter_version": self.exporter_version,
            "max_seq_len": self.max_seq_len,
            "modality": self.modality,
            "model_family": self.model_family,
            "model_id": self.model_id,
            "normalization": self.normalization,
            "output_dim": self.output_dim,
            "output_shape": list(self.output_shape) if self.output_shape is not None else None,
            "pooling": self.pooling.value,
            "preprocessor": self.preprocessor.to_dict(),
            "quantization": self.quantization,
            "source": self.source,
            "task": self.task,
            "tokenizer_files": list(self.tokenizer_files),
        }


ModelSpec = ModelContract
RerankerSpec = ModelContract
