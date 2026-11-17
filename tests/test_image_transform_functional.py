"""Pixel-level checks for the image transform primitives on real Pillow images."""

from typing import cast

import numpy as np
import pytest
from numpy.typing import NDArray

# Pillow ships in the optional [image] extra. The integration job installs only the dev
# group and still collects every module, so this module skips there instead of failing.
pytest.importorskip("PIL.Image", reason="Pillow belongs to the optional [image] extra")

from PIL import Image

from fastretrieval.common.types import NumpyArray
from fastretrieval.models.image.transform.functional import (
    center_crop,
    convert_to_rgb,
    crop_ndarray,
    normalize,
    pad2square,
    pil2ndarray,
    rescale,
    resize,
    resize_longest_edge,
    resize_ndarray,
)


def _pixels(array: NDArray[np.uint8]) -> NumpyArray:
    # NumpyArray omits uint8, yet the transforms take the uint8 pixel arrays Pillow produces.
    return cast(NumpyArray, array)


# --- convert_to_rgb ---------------------------------------------------------


def test_convert_to_rgb_returns_an_rgb_image_unchanged():
    image = Image.new("RGB", (2, 2), (10, 20, 30))

    assert convert_to_rgb(image) is image


def test_convert_to_rgb_expands_grayscale_into_three_equal_channels():
    converted = convert_to_rgb(Image.new("L", (2, 3), 77))

    assert converted.mode == "RGB"
    assert converted.size == (2, 3)
    np.testing.assert_array_equal(np.asarray(converted), np.full((3, 2, 3), 77, dtype=np.uint8))


def test_convert_to_rgb_drops_the_alpha_channel():
    converted = convert_to_rgb(Image.new("RGBA", (1, 1), (1, 2, 3, 0)))

    assert converted.getpixel((0, 0)) == (1, 2, 3)


# --- center_crop ------------------------------------------------------------


def test_center_crop_takes_the_middle_window_of_a_chw_array():
    image = np.arange(3 * 6 * 8).reshape(3, 6, 8)

    cropped = center_crop(image, (4, 4))

    np.testing.assert_array_equal(cropped, image[:, 1:5, 2:6])


def test_center_crop_turns_a_pil_image_into_chw_before_cropping():
    pixels = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)

    cropped = center_crop(Image.fromarray(pixels), (2, 4))

    assert cropped.shape == (3, 2, 4)
    np.testing.assert_array_equal(cropped, pixels.transpose(2, 0, 1)[:, 2:4, 2:6])


def test_center_crop_pads_with_zeros_when_the_crop_exceeds_the_image():
    image = np.full((3, 2, 2), 5, dtype=np.uint8)

    cropped = center_crop(_pixels(image), (4, 4))

    expected = np.zeros((3, 4, 4), dtype=np.float32)
    expected[:, 1:3, 1:3] = 5
    assert cropped.dtype == np.float32
    np.testing.assert_array_equal(cropped, expected)


def test_center_crop_pads_the_short_axis_and_crops_the_long_one():
    image = np.arange(3 * 2 * 6, dtype=np.float32).reshape(3, 2, 6)

    cropped = center_crop(image, (4, 2))

    assert cropped.shape == (3, 4, 2)
    np.testing.assert_array_equal(cropped[:, 0], 0)
    np.testing.assert_array_equal(cropped[:, 3], 0)
    np.testing.assert_array_equal(cropped[:, 1:3], image[:, :, 2:4])


# --- normalize --------------------------------------------------------------


def test_normalize_subtracts_and_divides_per_channel_of_a_chw_array():
    image = np.stack([np.full((2, 2), value, dtype=np.float32) for value in (1.0, 2.0, 3.0)])

    out = normalize(image, mean=[1.0, 1.0, 1.0], std=[1.0, 0.5, 2.0])

    assert out.dtype == np.float32
    np.testing.assert_allclose(out[:, 0, 0], [0.0, 2.0, 1.0])
    np.testing.assert_allclose(out[:, 1, 1], [0.0, 2.0, 1.0])


def test_normalize_leaves_a_float32_input_untouched():
    image = np.ones((3, 1, 1), dtype=np.float32)

    out = normalize(image, mean=0.5, std=0.5)

    assert out is not image
    np.testing.assert_allclose(out, 1.0)
    np.testing.assert_array_equal(image, 1.0)


def test_normalize_casts_an_integer_image_to_float32():
    image = np.full((3, 1, 1), 255, dtype=np.uint8)

    out = normalize(_pixels(image), mean=[0.0, 55.0, 255.0], std=[255.0, 100.0, 1.0])

    assert out.dtype == np.float32
    np.testing.assert_allclose(out.ravel(), [1.0, 2.0, 0.0])


def test_normalize_casts_a_float64_image_to_float32_without_mutating_it():
    image = np.full((3, 1, 1), 0.75, dtype=np.float64)

    out = normalize(image, mean=0.25, std=0.5)

    assert out.dtype == np.float32
    np.testing.assert_allclose(out, 1.0)
    np.testing.assert_array_equal(image, 0.75)


def test_normalize_reads_channels_from_axis_one_of_a_batch():
    batch = np.zeros((2, 3, 1, 1), dtype=np.float32)

    out = normalize(batch, mean=[1.0, 2.0, 3.0], std=[1.0, 1.0, 1.0])

    np.testing.assert_allclose(out[:, :, 0, 0], [[-1.0, -2.0, -3.0], [-1.0, -2.0, -3.0]])


def test_normalize_rejects_a_mean_with_the_wrong_channel_count():
    image = np.zeros((3, 1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match=r"mean must have the same number of channels.*got 2"):
        normalize(image, mean=[0.0, 0.0], std=1.0)


def test_normalize_rejects_a_std_with_the_wrong_channel_count():
    image = np.zeros((3, 1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match=r"std must have the same number of channels.*got 4"):
        normalize(image, mean=0.0, std=[1.0, 1.0, 1.0, 1.0])


# --- resize -----------------------------------------------------------------


def test_resize_passes_a_tuple_to_pillow_as_width_and_height():
    assert resize(Image.new("RGB", (8, 8)), (4, 2)).size == (4, 2)


def test_resize_with_an_int_scales_the_shorter_edge_of_a_portrait_image():
    assert resize(Image.new("RGB", (4, 8)), 2).size == (2, 4)


def test_resize_with_an_int_scales_the_shorter_edge_of_a_landscape_image():
    assert resize(Image.new("RGB", (9, 3)), 2).size == (6, 2)


def test_resize_applies_the_requested_resampling_filter():
    black_white = Image.fromarray(np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8))

    nearest = np.asarray(resize(black_white, (4, 1), resample=Image.Resampling.NEAREST))
    bilinear = np.asarray(resize(black_white, (4, 1)))

    np.testing.assert_array_equal(nearest[0, :, 0], [0, 0, 255, 255])
    assert ((bilinear[0, :, 0] > 0) & (bilinear[0, :, 0] < 255)).any()


# --- rescale / pil2ndarray --------------------------------------------------


def test_rescale_multiplies_and_casts_to_float32_by_default():
    out = rescale(_pixels(np.array([0, 51, 255], dtype=np.uint8)), 1 / 255)

    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [0.0, 0.2, 1.0], rtol=1e-6)


def test_rescale_honours_the_requested_dtype():
    out = rescale(np.array([2]), 0.5, dtype=np.float64)

    assert out.dtype == np.float64
    np.testing.assert_array_equal(out, [1.0])


def test_pil2ndarray_returns_channel_first_pixels():
    pixels = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)

    out = pil2ndarray(Image.fromarray(pixels))

    assert out.shape == (3, 2, 3)
    np.testing.assert_array_equal(out, pixels.transpose(2, 0, 1))


def test_pil2ndarray_passes_an_array_through():
    array = np.zeros((3, 1, 1))

    assert pil2ndarray(array) is array


# --- pad2square -------------------------------------------------------------


def test_pad2square_pastes_a_small_image_top_left_on_the_fill_colour():
    out = pad2square(Image.new("RGB", (2, 1), (255, 0, 0)), 3, fill_color=(0, 0, 255))

    pixels = np.asarray(out)
    assert out.size == (3, 3)
    np.testing.assert_array_equal(pixels[0, :2], [[255, 0, 0], [255, 0, 0]])
    np.testing.assert_array_equal(pixels[0, 2], [0, 0, 255])
    np.testing.assert_array_equal(pixels[1:], np.broadcast_to([0, 0, 255], (2, 3, 3)))


def test_pad2square_center_crops_a_wide_image_onto_black():
    pixels = np.zeros((1, 6, 3), dtype=np.uint8)
    pixels[0, :, 0] = [0, 10, 20, 30, 40, 50]

    out = np.asarray(pad2square(Image.fromarray(pixels), 4))

    np.testing.assert_array_equal(out[0, :, 0], [10, 20, 30, 40])
    np.testing.assert_array_equal(out[1:], 0)


def test_pad2square_center_crops_a_tall_image_onto_black():
    pixels = np.zeros((6, 1, 3), dtype=np.uint8)
    pixels[:, 0, 1] = [0, 10, 20, 30, 40, 50]

    out = np.asarray(pad2square(Image.fromarray(pixels), 4))

    np.testing.assert_array_equal(out[:, 0, 1], [10, 20, 30, 40])
    np.testing.assert_array_equal(out[:, 1:], 0)


# --- resize_longest_edge ----------------------------------------------------


@pytest.mark.parametrize(
    ("size", "max_size", "expected"),
    [
        ((8, 4), 6, (6, 4)),  # width longer, odd height rounded up
        ((8, 4), 5, (6, 2)),  # width longer, odd width rounded up
        ((4, 8), 6, (4, 6)),  # height longer, odd width rounded up
        ((4, 8), 5, (2, 6)),  # height longer, odd height rounded up
    ],
)
def test_resize_longest_edge_keeps_the_aspect_and_rounds_up_to_even(size, max_size, expected):
    assert resize_longest_edge(Image.new("RGB", size), max_size).size == expected


# --- crop_ndarray / resize_ndarray ------------------------------------------


def test_crop_ndarray_slices_a_channel_first_array():
    image = np.arange(3 * 4 * 5).reshape(3, 4, 5)

    np.testing.assert_array_equal(crop_ndarray(image, 1, 0, 3, 2), image[:, 0:2, 1:3])


def test_crop_ndarray_slices_a_channel_last_array():
    image = np.arange(4 * 5 * 3).reshape(4, 5, 3)

    out = crop_ndarray(image, 1, 0, 3, 2, channel_first=False)

    np.testing.assert_array_equal(out, image[0:2, 1:3, :])


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_resize_ndarray_round_trips_normalized_floats_through_uint8(dtype):
    image = np.full((3, 2, 2), 0.5, dtype=dtype)

    out = resize_ndarray(image, (4, 2))

    assert out.shape == (3, 2, 4)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, 127 / 255)


def test_resize_ndarray_keeps_a_uint8_channel_last_array_in_place():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[..., 2] = 200

    out = resize_ndarray(_pixels(image), (3, 5), channel_first=False)

    assert out.shape == (5, 3, 3)
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[..., 2], 200)
    np.testing.assert_array_equal(out[..., :2], 0)
