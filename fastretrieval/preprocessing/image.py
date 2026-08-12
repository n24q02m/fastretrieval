"""Biến ảnh thành tensor theo đúng khai báo trong hợp đồng model.

Không đoán tham số. Sai một bước chuẩn hoá có thể làm embedding sai mà không
gây lỗi, nên mọi giá trị đã khai báo đều lấy từ :class:`PreprocessorSpec`.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

from fastretrieval.contract.preprocessor import PreprocessorSpec

DEFAULT_SIZE = (224, 224)
DEFAULT_MEAN = (0.485, 0.456, 0.406)
DEFAULT_STD = (0.229, 0.224, 0.225)


def preprocess_images(images: Iterable[object], spec: PreprocessorSpec) -> np.ndarray:
    """Trả tensor NCHW ``float32`` đã chuẩn hoá theo ``spec``.

    ``image_size`` dùng quy ước ``(height, width)`` như hợp đồng; Pillow nhận
    kích thước theo thứ tự ``(width, height)`` nên phép đổi trục được thực hiện
    tại đúng ranh giới này.
    """
    if spec.kind != "image":
        raise ValueError(f"preprocess_images needs a spec with kind='image', got {spec.kind!r}")
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError(
            "image support needs pillow. Install it with: pip install 'fastretrieval[image]'"
        ) from exc

    height, width = spec.image_size or DEFAULT_SIZE
    mean = np.asarray(spec.image_mean or DEFAULT_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(spec.image_std or DEFAULT_STD, dtype=np.float32).reshape(3, 1, 1)

    batch: list[np.ndarray] = []
    for item in images:
        convert = getattr(item, "convert", None)
        if callable(convert):
            image = convert("RGB")
        else:
            with Image.open(Path(str(item))) as opened:
                image = opened.convert("RGB")
        resized = image.resize((width, height), Image.Resampling.BICUBIC)
        array = np.asarray(resized, dtype=np.float32).transpose(2, 0, 1) / 255.0
        batch.append((array - mean) / std)

    return np.stack(batch).astype(np.float32)
