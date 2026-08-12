"""Ổn định hóa manifest artifact dùng chung cho runtime và converter."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from fastretrieval.contract.model import ModelContract
from fastretrieval.contract.preprocessor import PreprocessorSpec

SCHEMA_VERSION = 1
_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "model_id",
        "source",
        "model_family",
        "task",
        "modality",
        "output_dim",
        "output_shape",
        "pooling",
        "normalization",
        "max_seq_len",
        "preprocessor",
        "tokenizer_files",
        "artifact_formats",
        "quantization",
        "exporter_version",
    }
)


def write_manifest(
    path: Path,
    contract: ModelContract,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Ghi manifest ổn định và nguyên tử.

    ``license`` và ``provenance`` chỉ xuất hiện khi caller cung cấp trong
    metadata; hàm không tự gán giá trị pháp lý cho artifact không rõ nguồn.
    """
    payload = {"schema_version": SCHEMA_VERSION, **contract.to_manifest_dict()}
    for key, value in (metadata or {}).items():
        if key in payload and key not in {"quantization", "exporter_version"}:
            raise ValueError(f"metadata key conflicts with manifest field: {key}")
        payload[key] = value
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_manifest(path: Path) -> ModelContract:
    """Đọc và validate manifest, fail-closed khi thiếu metadata bắt buộc."""
    manifest_path = Path(path)
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"manifest {manifest_path} must be a JSON object")
    model_id = payload.get("model_id", "<unknown>")
    missing = sorted(field for field in _REQUIRED_FIELDS if field not in payload)
    if missing:
        raise ValueError(f"{model_id}: manifest is missing {', '.join(missing)}")
    schema_version = payload["schema_version"]
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise ValueError(f"{model_id}: schema_version must be an integer")
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f"{model_id}: unsupported manifest schema_version {schema_version!r}")

    _validate_manifest_types(payload, model_id)
    try:
        preprocessor = PreprocessorSpec.from_dict(payload["preprocessor"])
        return ModelContract(
            model_id=payload["model_id"],
            source=payload["source"],
            task=payload["task"],
            modality=payload["modality"],
            model_family=payload["model_family"],
            output_dim=payload["output_dim"],
            output_shape=(
                tuple(payload["output_shape"]) if payload["output_shape"] is not None else None
            ),
            pooling=payload["pooling"],
            normalization=payload["normalization"],
            max_seq_len=payload["max_seq_len"],
            preprocessor=preprocessor,
            artifact_formats=tuple(payload["artifact_formats"]),
            tokenizer_files=tuple(payload["tokenizer_files"]),
            quantization=payload["quantization"],
            exporter_version=payload["exporter_version"],
        )
    except (AttributeError, TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"{model_id}: invalid manifest fields: {exc}") from exc


def _validate_manifest_types(payload: dict[str, Any], model_id: Any) -> None:
    """Validate JSON shapes before constructing ``ModelContract``.

    The explicit checks keep malformed JSON from surfacing as an unrelated
    ``AttributeError`` (for example ``None.strip()``) and reject booleans in
    integer positions, where Python otherwise treats them as integers.
    """

    string_fields = (
        "model_id",
        "source",
        "model_family",
        "task",
        "modality",
        "pooling",
    )
    for field in string_fields:
        if not isinstance(payload[field], str):
            raise ValueError(f"{model_id}: manifest field {field} must be a string")

    for field in ("output_dim", "max_seq_len"):
        value = payload[field]
        if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
            raise ValueError(f"{model_id}: manifest field {field} must be an integer or null")

    output_shape = payload["output_shape"]
    if output_shape is not None:
        if not isinstance(output_shape, list):
            raise ValueError(f"{model_id}: manifest field output_shape must be a list or null")
        for value in output_shape:
            if value is not None and (not isinstance(value, int) or isinstance(value, bool)):
                raise ValueError(
                    f"{model_id}: manifest output_shape values must be integers or null"
                )

    if not isinstance(payload["normalization"], bool):
        raise ValueError(f"{model_id}: manifest field normalization must be boolean")

    for field in ("tokenizer_files", "artifact_formats"):
        value = payload[field]
        if not isinstance(value, list) or any(
            not isinstance(item, str) or not item for item in value
        ):
            raise ValueError(f"{model_id}: manifest field {field} must be a list of strings")

    for field in ("quantization", "exporter_version"):
        value = payload[field]
        if value is not None and not isinstance(value, str):
            raise ValueError(f"{model_id}: manifest field {field} must be a string or null")
