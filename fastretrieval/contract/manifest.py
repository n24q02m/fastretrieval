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
    try:
        os.replace(temporary, destination)
    finally:
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
    if payload.get("schema_version", SCHEMA_VERSION) != SCHEMA_VERSION:
        raise ValueError(
            f"{model_id}: unsupported manifest schema_version {payload.get('schema_version')!r}"
        )
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
    except (TypeError, ValueError, KeyError) as exc:
        raise ValueError(f"{model_id}: invalid manifest fields: {exc}") from exc
