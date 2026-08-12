"""Architecture support matrix and fail-closed model profile resolution."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from fastretrieval.common.model_description import PoolingType
from fastretrieval.contract import (
    ModelContract,
    ModelModality,
    ModelTask,
    PreprocessorKind,
    PreprocessorSpec,
)

_TASK_ALIASES = {
    "feature-extraction": "dense",
    "embedding": "dense",
    "dense": "dense",
    "text-classification": "cross_encoder",
    "sequence-classification": "cross_encoder",
    "cross_encoder": "cross_encoder",
    "generative-reranker": "generative_reranker",
    "generative_reranker": "generative_reranker",
}
_MODALITIES = frozenset({"text", "image", "audio", "video"})


@dataclass(frozen=True)
class _SupportSpec:
    model_family: str
    architecture_keys: frozenset[str]
    tasks: frozenset[str]
    modalities: frozenset[str]
    artifact_formats: tuple[str, ...]
    pooling_modes: frozenset[str]
    normalization_modes: frozenset[bool]


@dataclass(frozen=True)
class ModelProfile:
    """Resolved model metadata from a supported architecture configuration."""

    source: str
    model_family: str
    architecture: str
    task: ModelTask
    modality: ModelModality
    config: Mapping[str, Any]
    output_dim: int | None
    max_seq_len: int | None
    tokenizer_files: tuple[str, ...]
    preprocessor: PreprocessorSpec
    artifact_formats: tuple[str, ...]
    input_names: tuple[str, ...]
    supported_tasks: frozenset[str]
    supported_modalities: frozenset[str]
    pooling_modes: frozenset[str]
    normalization_modes: frozenset[bool]

    def build_contract(
        self,
        *,
        pooling: str | PoolingType | None,
        normalization: bool | None,
        output_dim: int | None = None,
        output_shape: tuple[int | None, ...] | None = None,
        tokenizer_files: tuple[str, ...] | None = None,
        artifact_formats: tuple[str, ...] | None = None,
        preprocessor: PreprocessorSpec | None = None,
        quantization: str | None = None,
        exporter_version: str | None = None,
        yes_no: tuple[str, str] | None = None,
    ) -> ModelContract:
        """Build a complete contract without guessing missing behavior metadata."""
        context = self._context
        if pooling is None:
            raise ValueError(f"{context}: pooling metadata is required")
        if normalization is None:
            raise ValueError(f"{context}: normalization metadata is required")
        if not isinstance(normalization, bool):
            raise ValueError(f"{context}: normalization must be boolean")

        normalized_pooling = _normalize_pooling(pooling, context)
        if normalized_pooling.value not in self.pooling_modes:
            raise ValueError(
                f"{context}: pooling {normalized_pooling.value!r} is not supported; "
                f"pick from {sorted(self.pooling_modes)}"
            )
        if normalization not in self.normalization_modes:
            raise ValueError(
                f"{context}: normalization={normalization!r} is not supported; "
                f"pick from {sorted(self.normalization_modes)}"
            )

        if preprocessor is not None and not isinstance(preprocessor, PreprocessorSpec):
            raise ValueError(f"{context}: preprocessor must be a PreprocessorSpec")
        dim = output_dim if output_dim is not None else self.output_dim
        shape = output_shape
        if yes_no is not None:
            if self.task != "generative_reranker":
                raise ValueError(
                    f"{context}: yes_no output is only supported for generative_reranker"
                )
            if normalization:
                raise ValueError(f"{context}: yes_no logits cannot declare normalization=True")
            dim = 2
            shape = (2,)
        elif (
            self.task in {"cross_encoder", "generative_reranker"} and dim is None and shape is None
        ):
            raise ValueError(
                f"{context}: task {self.task!r} requires explicit output_dim/output_shape "
                "or a yes_no head"
            )
        if dim is None and shape is None:
            raise ValueError(f"{context}: output_dim or output_shape metadata is required")
        if shape is None and dim is not None:
            shape = (dim,)
        if dim is None and shape is not None and len(shape) == 1:
            dim = shape[0]

        files = self.tokenizer_files if tokenizer_files is None else tuple(tokenizer_files)
        formats = self.artifact_formats if artifact_formats is None else tuple(artifact_formats)
        if not files:
            raise ValueError(f"{context}: tokenizer_files metadata is required")
        if not formats:
            raise ValueError(f"{context}: artifact_formats metadata is required")
        if len(set(formats)) != len(formats):
            raise ValueError(f"{context}: artifact_formats must not contain duplicates")
        unsupported_formats = [item for item in formats if item not in self.artifact_formats]
        if unsupported_formats:
            raise ValueError(
                f"{context}: artifact format(s) are outside the profile: "
                f"{unsupported_formats!r}; pick from {self.artifact_formats!r}"
            )
        processor = self.preprocessor if preprocessor is None else preprocessor
        if processor.kind != self.modality:
            raise ValueError(
                f"{context}: preprocessor kind {processor.kind!r} does not match "
                f"modality {self.modality!r}"
            )

        return ModelContract(
            model_id=str(self.config.get("_model_id", self.source)),
            source=self.source,
            task=self.task,
            modality=self.modality,
            model_family=self.model_family,
            output_dim=dim,
            output_shape=shape,
            pooling=normalized_pooling,
            normalization=normalization,
            max_seq_len=self.max_seq_len,
            preprocessor=processor,
            artifact_formats=formats,
            tokenizer_files=files,
            quantization=quantization,
            exporter_version=exporter_version,
        )

    @property
    def _context(self) -> str:
        return f"source={self.source} task={self.task} modality={self.modality}"


SUPPORT_MATRIX: tuple[_SupportSpec, ...] = (
    _SupportSpec(
        model_family="qwen3",
        architecture_keys=frozenset({"qwen3", "qwen3model", "qwen3forcausallm"}),
        tasks=frozenset({"dense", "generative_reranker"}),
        modalities=frozenset({"text"}),
        artifact_formats=("onnx", "gguf"),
        pooling_modes=frozenset({"CLS", "MEAN", "LAST_TOKEN"}),
        normalization_modes=frozenset({True, False}),
    ),
    _SupportSpec(
        model_family="bert",
        architecture_keys=frozenset({"bert", "bertmodel", "bertformodel", "bertformaskedlm"}),
        tasks=frozenset({"dense", "cross_encoder"}),
        modalities=frozenset({"text"}),
        artifact_formats=("onnx",),
        pooling_modes=frozenset({"CLS", "MEAN", "LAST_TOKEN"}),
        normalization_modes=frozenset({True, False}),
    ),
)


def resolve_profile(
    source: str | Path,
    *,
    task: str,
    modality: str,
) -> ModelProfile:
    """Resolve a local or HuggingFace model through the explicit support matrix."""
    source_text = str(source)
    normalized_task = _normalize_task(task, source_text, modality)
    if modality not in _MODALITIES:
        raise ValueError(
            f"source={source_text} task={task} modality={modality}: unsupported modality"
        )

    config = _load_config(source_text, source)
    declared = _declared_architectures(config)
    context = f"source={source_text} task={task} modality={modality}"
    matches = [
        candidate
        for candidate in SUPPORT_MATRIX
        if declared.intersection(candidate.architecture_keys)
    ]
    if not matches:
        raise ValueError(
            f"{context}: architecture {sorted(declared) or '<missing>'!r} is outside support matrix"
        )
    if len(matches) != 1:
        families = ", ".join(sorted(candidate.model_family for candidate in matches))
        raise ValueError(
            f"{context}: architecture {sorted(declared)!r} matches multiple support profiles "
            f"({families}); declare exactly one supported family"
        )
    spec = matches[0]
    if normalized_task not in spec.tasks:
        raise ValueError(f"{context}: task is not supported for {spec.model_family!r}")
    if modality not in spec.modalities:
        raise ValueError(f"{context}: modality is not supported for {spec.model_family!r}")

    model_type = config.get("model_type")
    if isinstance(model_type, str) and model_type.lower() in spec.architecture_keys:
        architecture = model_type.lower()
    else:
        architecture = sorted(declared.intersection(spec.architecture_keys))[0]
    model_family = config.get("fastretrieval_model_family", spec.model_family)
    if not isinstance(model_family, str) or not model_family.strip():
        raise ValueError(f"{context}: fastretrieval_model_family must be a non-empty string")

    output_dim = None
    if normalized_task == "dense":
        output_dim = _read_positive_int(config, "sentence_embedding_dimension")
        if output_dim is None:
            output_dim = _read_positive_int(config, "projection_dim")
        if output_dim is None:
            output_dim = _read_positive_int(config, "hidden_size")
    max_seq_len = _read_positive_int(config, "max_seq_len")
    if max_seq_len is None:
        max_seq_len = _read_positive_int(config, "max_position_embeddings")

    processor = PreprocessorSpec(
        kind=cast(PreprocessorKind, modality), config_file="preprocessor_config.json"
    )
    local_dir = _local_model_dir(source)
    if local_dir is not None:
        discovered = PreprocessorSpec.from_model_dir(local_dir)
        if discovered is not None:
            processor = discovered

    config_with_id = dict(config)
    config_with_id["_model_id"] = source_text
    input_names = _read_input_names(config, context)
    return ModelProfile(
        source=source_text,
        model_family=model_family,
        architecture=architecture,
        task=cast(ModelTask, normalized_task),
        modality=cast(ModelModality, modality),
        config=config_with_id,
        output_dim=output_dim,
        max_seq_len=max_seq_len,
        tokenizer_files=("config.json", "tokenizer.json", "tokenizer_config.json"),
        preprocessor=processor,
        artifact_formats=spec.artifact_formats,
        input_names=input_names,
        supported_tasks=spec.tasks,
        supported_modalities=spec.modalities,
        pooling_modes=spec.pooling_modes,
        normalization_modes=spec.normalization_modes,
    )


def _normalize_task(task: str, source: str, modality: str) -> str:
    normalized = _TASK_ALIASES.get(task.lower())
    if normalized is None:
        raise ValueError(f"source={source} task={task} modality={modality}: unsupported task")
    return normalized


def _normalize_pooling(value: str | PoolingType, context: str) -> PoolingType:
    if isinstance(value, PoolingType):
        return value
    normalized = str(value).upper()
    try:
        return PoolingType(normalized)
    except ValueError as exc:
        raise ValueError(f"{context}: unsupported pooling {value!r}") from exc


def _load_config(source_text: str, source: str | Path) -> dict[str, Any]:
    local_dir = _local_model_dir(source)
    if local_dir is not None:
        config_path = local_dir / "config.json"
        if not config_path.is_file():
            raise ValueError(
                f"source={source_text!r}: config.json is required for profile resolution"
            )
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"source={source_text!r}: cannot read config.json: {exc}") from exc
        if not isinstance(payload, dict):
            raise ValueError(f"source={source_text!r}: config.json must be a JSON object")
        return payload

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ValueError(
            f"source={source_text!r}: install huggingface-hub or provide a local model directory"
        ) from exc
    try:
        config_path = hf_hub_download(repo_id=source_text, filename="config.json")
        payload = json.loads(Path(config_path).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(
            f"source={source_text!r}: cannot inspect config.json for support-matrix resolution: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise ValueError(f"source={source_text!r}: config.json must be a JSON object")
    return payload


def _local_model_dir(source: str | Path) -> Path | None:
    try:
        path = Path(source)
    except (OSError, ValueError):
        return None
    if path.is_dir():
        return path
    if path.is_file() and path.name == "config.json":
        return path.parent
    return None


def _declared_architectures(config: Mapping[str, Any]) -> set[str]:
    declared: set[str] = set()
    model_type = config.get("model_type")
    if isinstance(model_type, str) and model_type:
        declared.add(model_type.lower())
    architectures = config.get("architectures", ())
    if isinstance(architectures, list):
        declared.update(
            value.lower() for value in architectures if isinstance(value, str) and value
        )
    return declared


def _read_positive_int(config: Mapping[str, Any], key: str) -> int | None:
    value = config.get(key)
    if isinstance(value, int) and not isinstance(value, bool) and value > 0:
        return value
    return None


def _read_input_names(config: Mapping[str, Any], context: str) -> tuple[str, ...]:
    value = config.get("input_names", ("input_ids", "attention_mask"))
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{context}: input_names must be a sequence of non-empty strings")
    names = tuple(value)
    if not names or any(not isinstance(name, str) or not name.strip() for name in names):
        raise ValueError(f"{context}: input_names must be a sequence of non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError(f"{context}: input_names must not contain duplicates")
    return names


__all__ = ["ModelProfile", "SUPPORT_MATRIX", "resolve_profile"]
