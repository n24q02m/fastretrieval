"""Khai báo tiền xử lý đầu vào cho model do người dùng tự cắm.

Hợp đồng đầu ra không đủ để mô tả model ảnh: sai một bước chuẩn hoá pixel có
thể tạo embedding sai mà không gây lỗi. Module này giữ khai báo đầu vào độc
lập với model family và không tải dependency nặng ở thời điểm import.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

PreprocessorKind = Literal["text", "image", "audio", "video", "none"]
_PREPROCESSOR_KINDS = frozenset({"text", "image", "audio", "video", "none"})
CONFIG_FILENAME = "preprocessor_config.json"


@dataclass(frozen=True)
class PreprocessorSpec:
    """Mô tả cách biến đầu vào thô thành tensor cho model."""

    kind: PreprocessorKind = "text"
    config_file: str | None = None
    image_size: tuple[int, int] | None = None
    image_mean: tuple[float, float, float] | None = None
    image_std: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        if self.kind not in _PREPROCESSOR_KINDS:
            raise ValueError(f"unsupported preprocessor kind: {self.kind!r}")
        if self.image_size is not None and (
            len(self.image_size) != 2 or any(value <= 0 for value in self.image_size)
        ):
            raise ValueError(f"image_size must contain two positive values: {self.image_size!r}")
        if self.image_mean is not None and len(self.image_mean) != 3:
            raise ValueError(f"image_mean must contain three channel values: {self.image_mean!r}")
        if self.image_std is not None and (
            len(self.image_std) != 3 or any(value == 0 for value in self.image_std)
        ):
            raise ValueError(f"image_std must contain three non-zero values: {self.image_std!r}")

    @classmethod
    def from_model_dir(cls, model_dir: Path) -> PreprocessorSpec | None:
        """Dựng spec từ ``preprocessor_config.json`` trong thư mục model."""
        path = Path(model_dir) / CONFIG_FILENAME
        if not path.is_file():
            return None
        try:
            raw: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path} is not valid JSON: {exc}") from exc
        return cls._from_hf_config(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> PreprocessorSpec:
        """Đọc spec đã được serialize trong manifest."""
        if not isinstance(raw, dict):
            raise ValueError(f"preprocessor must be an object, got {type(raw).__name__}")
        try:
            kind = raw["kind"]
        except KeyError as exc:
            raise ValueError("preprocessor is missing kind") from exc
        image_size = raw.get("image_size")
        if image_size is not None:
            image_size = tuple(int(value) for value in image_size)
        image_mean = raw.get("image_mean")
        if image_mean is not None:
            image_mean = tuple(float(value) for value in image_mean)
        image_std = raw.get("image_std")
        if image_std is not None:
            image_std = tuple(float(value) for value in image_std)
        return cls(
            kind=kind,
            config_file=raw.get("config_file"),
            image_size=image_size,
            image_mean=image_mean,
            image_std=image_std,
        )

    def to_dict(self) -> dict[str, Any]:
        """Trả representation JSON ổn định, không giữ tuple Python."""
        return {
            "config_file": self.config_file,
            "image_mean": list(self.image_mean) if self.image_mean is not None else None,
            "image_size": list(self.image_size) if self.image_size is not None else None,
            "image_std": list(self.image_std) if self.image_std is not None else None,
            "kind": self.kind,
        }

    @classmethod
    def _from_hf_config(cls, raw: dict[str, Any]) -> PreprocessorSpec:
        if "image_processor_type" not in raw and "image_mean" not in raw:
            return cls(kind="text", config_file=CONFIG_FILENAME)
        return cls(
            kind="image",
            config_file=CONFIG_FILENAME,
            image_size=_read_size(raw.get("size")),
            image_mean=_read_triple(raw.get("image_mean")),
            image_std=_read_triple(raw.get("image_std")),
        )


def _read_size(value: Any) -> tuple[int, int] | None:
    """Chuẩn hoá trường ``size`` của các processor Hugging Face."""
    if value is None:
        return None
    if isinstance(value, int) and not isinstance(value, bool):
        return (value, value)
    if isinstance(value, dict):
        if "height" in value and "width" in value:
            return (int(value["height"]), int(value["width"]))
        if "shortest_edge" in value:
            edge = int(value["shortest_edge"])
            return (edge, edge)
    raise ValueError(f"unsupported preprocessor size field: {value!r}")


def _read_triple(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    try:
        items = [float(item) for item in value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected three channel values, got {value!r}") from exc
    if len(items) != 3:
        raise ValueError(f"expected 3 channel values, got {len(items)}: {value!r}")
    return (items[0], items[1], items[2])
