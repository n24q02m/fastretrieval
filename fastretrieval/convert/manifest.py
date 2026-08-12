"""Converter-facing manifest helpers built on the runtime contract schema."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastretrieval.contract.manifest import load_manifest as _load_contract_manifest
from fastretrieval.contract.manifest import write_manifest as _write_contract_manifest
from fastretrieval.contract.model import ModelContract

MANIFEST_FILENAME = "fastretrieval-manifest.json"


def write_manifest(
    out_dir: str | Path,
    contract: ModelContract,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write the versioned manifest next to a converted artifact atomically."""
    destination = Path(out_dir) / MANIFEST_FILENAME
    contract_payload = contract.to_manifest_dict()
    extra: dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        if key in contract_payload:
            if contract_payload[key] != value:
                raise ValueError(f"metadata key conflicts with manifest field: {key}")
            continue
        extra[key] = value
    _write_contract_manifest(destination, contract, extra)
    return destination


def load_manifest(directory_or_path: str | Path) -> ModelContract:
    """Load a converter manifest from a directory or an explicit JSON path."""
    path = Path(directory_or_path)
    if path.is_dir():
        path /= MANIFEST_FILENAME
    return _load_contract_manifest(path)


__all__ = ["MANIFEST_FILENAME", "load_manifest", "write_manifest"]
