"""Numeric behaviour of ``preprocess_images`` on real Pillow images and files."""

import sys
from pathlib import Path
from typing import TypedDict

import numpy as np
import pytest

# Pillow ships in the optional [image] extra. The integration job installs only the dev
# group and still collects every module, so this module skips there instead of failing.
pytest.importorskip("PIL.Image", reason="Pillow belongs to the optional [image] extra")

from PIL import Image

from fastretrieval.contract import PreprocessorSpec
from fastretrieval.preprocessing.image import (
    DEFAULT_MEAN,
    DEFAULT_SIZE,
    DEFAULT_STD,
    preprocess_images,
)


class _Normalization(TypedDict):
    image_mean: tuple[float, float, float]
    image_std: tuple[float, float, float]


IDENTITY: _Normalization = {"image_mean": (0.0, 0.0, 0.0), "image_std": (1.0, 1.0, 1.0)}


def _save(path: Path, color: tuple[int, int, int]) -> Path:
    Image.new("RGB", (3, 3), color).save(path)
    return path


def test_files_are_opened_from_str_and_path_inputs(tmp_path):
    red = _save(tmp_path / "red.png", (255, 0, 0))
    blue = _save(tmp_path / "blue.png", (0, 0, 255))
    spec = PreprocessorSpec(kind="image", image_size=(2, 2), **IDENTITY)

    out = preprocess_images([str(red), blue], spec)

    assert out.shape == (2, 3, 2, 2)
    np.testing.assert_allclose(out[0, :, 0, 0], [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(out[1, :, 1, 1], [0.0, 0.0, 1.0], atol=1e-6)


def test_image_size_is_read_as_height_then_width():
    spec = PreprocessorSpec(kind="image", image_size=(2, 5))

    out = preprocess_images([Image.new("RGB", (7, 7))], spec)

    assert out.shape == (1, 3, 2, 5)


def test_bare_spec_uses_default_size_and_imagenet_statistics():
    out = preprocess_images(
        [Image.new("RGB", (8, 8), (255, 0, 51))], PreprocessorSpec(kind="image")
    )

    assert out.shape == (1, 3, *DEFAULT_SIZE)
    assert out.dtype == np.float32
    pixel = np.array([1.0, 0.0, 0.2])
    expected = (pixel - np.array(DEFAULT_MEAN)) / np.array(DEFAULT_STD)
    np.testing.assert_allclose(out[0, :, 0, 0], expected, rtol=1e-5)
    np.testing.assert_allclose(out[0, :, -1, -1], expected, rtol=1e-5)


def test_grayscale_and_alpha_images_are_converted_to_rgb():
    spec = PreprocessorSpec(kind="image", image_size=(1, 1), **IDENTITY)

    out = preprocess_images(
        [Image.new("L", (2, 2), 51), Image.new("RGBA", (2, 2), (0, 255, 0, 0))], spec
    )

    np.testing.assert_allclose(out[0, :, 0, 0], [0.2, 0.2, 0.2], atol=1e-6)
    np.testing.assert_allclose(out[1, :, 0, 0], [0.0, 1.0, 0.0], atol=1e-6)


def test_custom_mean_and_std_are_applied_per_channel():
    spec = PreprocessorSpec(
        kind="image", image_size=(1, 1), image_mean=(0.5, 0.0, 1.0), image_std=(0.5, 0.25, 2.0)
    )

    out = preprocess_images([Image.new("RGB", (1, 1), (255, 255, 255))], spec)

    np.testing.assert_allclose(out[0, :, 0, 0], [1.0, 4.0, 0.0], atol=1e-5)


def test_generator_input_is_consumed():
    spec = PreprocessorSpec(kind="image", image_size=(1, 1))

    out = preprocess_images((Image.new("RGB", (1, 1)) for _ in range(3)), spec)

    assert out.shape == (3, 3, 1, 1)


def test_empty_batch_returns_an_empty_tensor_of_the_declared_size():
    out = preprocess_images([], PreprocessorSpec(kind="image", image_size=(2, 5)))

    assert out.shape == (0, 3, 2, 5)
    assert out.dtype == np.float32


@pytest.mark.parametrize("kind", ["text", "none", "audio"])
def test_non_image_specs_are_rejected(kind):
    with pytest.raises(ValueError, match=f"needs a spec with kind='image', got '{kind}'"):
        preprocess_images([], PreprocessorSpec(kind=kind))


def test_missing_pillow_raises_an_install_hint(monkeypatch):
    # Optional-dependency boundary: hide Pillow the way an install without the image extra would.
    monkeypatch.setitem(sys.modules, "PIL", None)

    with pytest.raises(ImportError, match=r"pip install 'fastretrieval\[image\]'") as info:
        preprocess_images([], PreprocessorSpec(kind="image"))

    assert isinstance(info.value.__cause__, ImportError)
