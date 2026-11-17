"""Behaviour of the image transform operators and the config-driven ``Compose`` pipeline."""

from typing import TypeVar

import numpy as np
import pytest

# Pillow ships in the optional [image] extra. The integration job installs only the dev
# group and still collects every module, so this module skips there instead of failing.
pytest.importorskip("PIL.Image", reason="Pillow belongs to the optional [image] extra")

from PIL import Image

from fastretrieval.models.image import transform as transform_package
from fastretrieval.models.image.transform.operators import (
    CenterCrop,
    Compose,
    ConvertToRGB,
    ImageSplitter,
    Normalize,
    PadtoSquare,
    PILtoNDarray,
    Rescale,
    Resize,
    ResizeForVisionEncoder,
    ResizeLongestEdge,
    SquareResize,
    Transform,
)

HALF = [0.5, 0.5, 0.5]
_T = TypeVar("_T", bound=Transform)


def _types(compose: Compose) -> list[type]:
    return [type(step) for step in compose.transforms]


def _step(compose: Compose, index: int, kind: type[_T]) -> _T:
    step = compose.transforms[index]
    assert isinstance(step, kind)
    return step


# --- individual operators ---------------------------------------------------


def test_transform_base_class_requires_an_override():
    with pytest.raises(NotImplementedError, match="Subclasses must implement this method"):
        Transform()([])


def test_package_reexports_compose():
    assert transform_package.Compose is Compose


def test_convert_to_rgb_operator_converts_every_image():
    out = ConvertToRGB()([Image.new("L", (1, 1), 9), Image.new("RGBA", (1, 1), (1, 2, 3, 4))])

    assert [image.mode for image in out] == ["RGB", "RGB"]
    assert [image.getpixel((0, 0)) for image in out] == [(9, 9, 9), (1, 2, 3)]


def test_center_crop_operator_crops_every_image_to_height_and_width():
    image = np.arange(3 * 4 * 6).reshape(3, 4, 6)

    (out,) = CenterCrop((2, 2))([image])

    np.testing.assert_array_equal(out, image[:, 1:3, 2:4])


def test_resize_operator_scales_the_short_edge_with_its_filter():
    op = Resize(3, resample=Image.Resampling.NEAREST)

    out = op([Image.new("RGB", (6, 12)), Image.new("RGB", (12, 6))])

    assert [image.size for image in out] == [(3, 6), (6, 3)]


def test_resize_operator_defaults_to_bicubic():
    assert Resize(4).resample == Image.Resampling.BICUBIC


def test_pil_to_ndarray_operator_returns_channel_first_arrays():
    (out,) = PILtoNDarray()([Image.new("RGB", (5, 2))])

    assert out.shape == (3, 2, 5)


def test_pad_to_square_operator_uses_its_size_and_fill_colour():
    (out,) = PadtoSquare(size=3, fill_color=(9, 8, 7))([Image.new("RGB", (1, 1), (1, 1, 1))])

    pixels = np.asarray(out)
    assert out.size == (3, 3)
    np.testing.assert_array_equal(pixels[0, 0], [1, 1, 1])
    np.testing.assert_array_equal(pixels[2, 2], [9, 8, 7])


def test_resize_longest_edge_operator_bounds_the_longest_side():
    op = ResizeLongestEdge(4)

    out = op([Image.new("RGB", (8, 4)), Image.new("RGB", (2, 8))])

    assert op.resample == Image.Resampling.LANCZOS
    assert [image.size for image in out] == [(4, 2), (2, 4)]


def test_rescale_operator_scales_a_flat_list():
    (out,) = Rescale()([np.array([255, 0], dtype=np.uint8)])

    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [1.0, 0.0])


def test_rescale_operator_scales_nested_patch_lists():
    out = Rescale(scale=0.5)([[np.array([2.0]), np.array([4.0])], [np.array([6.0])]])

    assert [[patch.tolist() for patch in patches] for patches in out] == [[[1.0], [2.0]], [[3.0]]]


def test_rescale_operator_returns_an_empty_list_for_no_images():
    assert Rescale()([]) == []


def test_normalize_operator_normalizes_a_flat_list():
    (out,) = Normalize(mean=HALF, std=HALF)([np.ones((3, 1, 1), dtype=np.float32)])

    np.testing.assert_allclose(out, 1.0)


def test_normalize_operator_normalizes_nested_patch_lists():
    patches = [[np.zeros((3, 1, 1), dtype=np.float32)], [np.ones((3, 1, 1), dtype=np.float32)]]

    out = Normalize(mean=0.5, std=0.5)(patches)

    np.testing.assert_allclose(out[0][0], -1.0)
    np.testing.assert_allclose(out[1][0], 1.0)


def test_normalize_operator_returns_an_empty_list_for_no_images():
    assert Normalize(mean=0.0, std=1.0)([]) == []


def test_resize_for_vision_encoder_rounds_a_landscape_image_up_to_the_grid():
    op = ResizeForVisionEncoder(4)

    (out,) = op([np.full((3, 3, 6), 0.25, dtype=np.float32)])

    assert op.resample == Image.Resampling.LANCZOS
    assert out.shape == (3, 4, 8)
    np.testing.assert_allclose(out, 63 / 255)


def test_resize_for_vision_encoder_rounds_a_portrait_image_up_to_the_grid():
    (out,) = ResizeForVisionEncoder(4)([np.zeros((3, 6, 3), dtype=np.uint8)])

    assert out.shape == (3, 8, 4)
    assert out.dtype == np.uint8


def test_image_splitter_emits_row_major_patches_and_a_global_view():
    image = np.arange(3 * 5 * 6, dtype=np.float32).reshape(3, 5, 6) / 100.0

    (frames,) = ImageSplitter(4)([image])

    assert len(frames) == 5
    np.testing.assert_array_equal(frames[0], image[:, 0:3, 0:3])
    np.testing.assert_array_equal(frames[1], image[:, 0:3, 3:6])
    np.testing.assert_array_equal(frames[2], image[:, 3:5, 0:3])
    np.testing.assert_array_equal(frames[3], image[:, 3:5, 3:6])
    assert frames[4].shape == (3, 4, 4)
    assert frames[4].dtype == np.float32


def test_image_splitter_keeps_a_small_image_as_a_single_frame():
    small = np.zeros((3, 4, 4), dtype=np.float32)

    out = ImageSplitter(4)([small, small])

    assert len(out) == 2
    assert out[0] == [small]
    assert out[0][0] is small


def test_square_resize_wraps_each_resized_image_in_its_own_list():
    out = SquareResize(4)([np.zeros((3, 2, 6), dtype=np.uint8), np.zeros((3, 8, 8), np.uint8)])

    assert [[frame.shape for frame in frames] for frames in out] == [[(3, 4, 4)], [(3, 4, 4)]]


def test_compose_applies_transforms_in_order():
    pipeline = Compose([Rescale(scale=2.0), Normalize(mean=1.0, std=1.0)])

    (out,) = pipeline([np.ones((3, 1, 1))])

    np.testing.assert_allclose(out, 1.0)


def test_compose_without_transforms_returns_the_input():
    images = [object()]

    assert Compose([])(images) is images


# --- Compose.from_config: CLIP / Siglip --------------------------------------


def test_clip_config_resizes_crops_rescales_and_normalizes():
    config = {
        "image_processor_type": "CLIPImageProcessor",
        "do_resize": True,
        "size": {"shortest_edge": 4},
        "do_center_crop": True,
        "crop_size": 4,
        "do_normalize": True,
        "image_mean": HALF,
        "image_std": HALF,
    }

    pipeline = Compose.from_config(config)
    (out,) = pipeline([Image.new("RGB", (8, 6), (0, 255, 255))])

    assert _types(pipeline) == [ConvertToRGB, Resize, CenterCrop, PILtoNDarray, Rescale, Normalize]
    assert _step(pipeline, 1, Resize).size == 4
    assert _step(pipeline, 1, Resize).resample == Image.Resampling.BICUBIC
    assert out.shape == (3, 4, 4)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out[:, 0, 0], [-1.0, 1.0, 1.0], atol=1e-6)


def test_clip_is_the_default_processor_and_skips_disabled_steps():
    pipeline = Compose.from_config({"do_rescale": False})

    (out,) = pipeline([Image.new("RGB", (3, 2), (5, 6, 7))])

    assert _types(pipeline) == [ConvertToRGB, PILtoNDarray]
    assert out.dtype == np.uint8
    np.testing.assert_array_equal(out[:, 1, 2], [5, 6, 7])


def test_siglip_config_resizes_to_an_explicit_square_size():
    config = {
        "image_processor_type": "SiglipImageProcessor",
        "do_resize": True,
        "size": {"height": 4, "width": 4},
        "resample": Image.Resampling.NEAREST,
        "rescale_factor": 0.5,
    }

    pipeline = Compose.from_config(config)
    (out,) = pipeline([Image.new("RGB", (6, 2), (2, 4, 6))])

    assert _types(pipeline) == [ConvertToRGB, Resize, PILtoNDarray, Rescale]
    assert _step(pipeline, 1, Resize).resample == Image.Resampling.NEAREST
    assert out.shape == (3, 4, 4)
    np.testing.assert_allclose(out[:, 3, 3], [1.0, 2.0, 3.0])


def test_clip_config_rejects_a_size_without_known_keys():
    config = {"do_resize": True, "size": {"longest_edge": 4}}

    with pytest.raises(ValueError, match="Size must contain either 'shortest_edge' or 'height'"):
        Compose.from_config(config)


def test_clip_center_crop_accepts_a_height_width_dict():
    config = {"do_center_crop": True, "crop_size": {"height": 2, "width": 3}, "do_rescale": False}

    (out,) = Compose.from_config(config)([Image.new("RGB", (5, 5))])

    assert out.shape == (3, 2, 3)


def test_clip_center_crop_rejects_a_list_crop_size():
    with pytest.raises(ValueError, match=r"Invalid crop size: \[4, 4\]"):
        Compose.from_config({"do_center_crop": True, "crop_size": [4, 4]})


def test_unknown_processor_type_is_rejected():
    with pytest.raises(ValueError, match="Preprocessor BeitImageProcessor is not supported"):
        Compose.from_config({"image_processor_type": "BeitImageProcessor"})


def test_center_crop_step_rejects_an_unknown_processor_type():
    with pytest.raises(ValueError, match="Preprocessor BeitImageProcessor is not supported"):
        Compose._get_center_crop([], {"image_processor_type": "BeitImageProcessor"})


# --- Compose.from_config: ConvNext -------------------------------------------


def test_convnext_small_edge_resizes_by_crop_pct_then_center_crops():
    config = {
        "image_processor_type": "ConvNextFeatureExtractor",
        "size": {"shortest_edge": 7},
        "crop_pct": 0.875,
        "do_rescale": False,
    }

    pipeline = Compose.from_config(config)
    (out,) = pipeline([Image.new("RGB", (16, 12), (3, 2, 1))])

    assert _types(pipeline) == [ConvertToRGB, Resize, CenterCrop, PILtoNDarray]
    assert _step(pipeline, 1, Resize).size == 8
    assert _step(pipeline, 2, CenterCrop).size == (7, 7)
    assert out.shape == (3, 7, 7)
    np.testing.assert_array_equal(out[:, 3, 3], [3, 2, 1])


def test_convnext_uses_the_default_crop_pct():
    config = {"image_processor_type": "ConvNextFeatureExtractor", "size": {"shortest_edge": 224}}

    pipeline = Compose.from_config(config)

    assert _step(pipeline, 1, Resize).size == 256


def test_convnext_large_edge_resizes_straight_to_a_square():
    config = {
        "image_processor_type": "ConvNextFeatureExtractor",
        "size": {"shortest_edge": 384},
        "do_rescale": False,
    }

    pipeline = Compose.from_config(config)
    (out,) = pipeline([Image.new("RGB", (2, 1))])

    assert _types(pipeline) == [ConvertToRGB, Resize, PILtoNDarray]
    assert _step(pipeline, 1, Resize).size == (384, 384)
    assert out.shape == (3, 384, 384)


def test_convnext_rejects_a_size_without_shortest_edge():
    config = {"image_processor_type": "ConvNextFeatureExtractor", "size": {"height": 4}}

    with pytest.raises(ValueError, match="must contain 'shortest_edge' key. Got dict_keys"):
        Compose.from_config(config)


# --- Compose.from_config: JinaCLIP -------------------------------------------


def test_jina_config_resizes_pads_and_normalizes_with_mean_std_keys():
    config = {
        "image_processor_type": "JinaCLIPImageProcessor",
        "size": 4,
        "interpolation": "BiCuBiC",
        "mean": HALF,
        "std": HALF,
    }

    pipeline = Compose.from_config(config)
    (out,) = pipeline([Image.new("RGB", (2, 4), (255, 0, 255))])

    expected_types = [ConvertToRGB, Resize, PadtoSquare, PILtoNDarray, Rescale, Normalize]
    assert _types(pipeline) == expected_types
    assert _step(pipeline, 1, Resize).resample == Image.Resampling.BICUBIC
    assert _step(pipeline, 2, PadtoSquare).fill_color == 0
    assert out.shape == (3, 4, 4)
    np.testing.assert_allclose(out[:, 1, 1], [1.0, -1.0, 1.0], atol=1e-6)


def test_jina_config_without_shortest_mode_only_pads_with_the_fill_colour():
    config = {
        "image_processor_type": "JinaCLIPImageProcessor",
        "size": 3,
        "resize_mode": "longest",
        "fill_color": (9, 9, 9),
        "do_rescale": False,
    }

    pipeline = Compose.from_config(config)
    (out,) = pipeline([Image.new("RGB", (1, 2), (1, 1, 1))])

    assert _types(pipeline) == [ConvertToRGB, PadtoSquare, PILtoNDarray]
    assert out.shape == (3, 3, 3)
    np.testing.assert_array_equal(out[:, 1, 0], [1, 1, 1])
    np.testing.assert_array_equal(out[:, 2, 2], [9, 9, 9])


def test_jina_config_accepts_an_integer_interpolation():
    config = {
        "image_processor_type": "JinaCLIPImageProcessor",
        "size": 4,
        "interpolation": int(Image.Resampling.BILINEAR),
    }

    pipeline = Compose.from_config(config)

    assert _step(pipeline, 1, Resize).resample == Image.Resampling.BILINEAR


def test_jina_config_defaults_to_bicubic_interpolation():
    pipeline = Compose.from_config({"image_processor_type": "JinaCLIPImageProcessor", "size": 4})

    assert _step(pipeline, 1, Resize).resample == Image.Resampling.BICUBIC


def test_jina_config_rejects_an_unknown_interpolation_name():
    config = {"image_processor_type": "JinaCLIPImageProcessor", "size": 4, "interpolation": "x"}

    with pytest.raises(ValueError, match="Unknown interpolation method: x"):
        Compose.from_config(config)


def test_jina_resize_step_adds_nothing_without_a_size():
    transforms: list[Transform] = []

    Compose._get_resize(transforms, {"image_processor_type": "JinaCLIPImageProcessor"})

    assert transforms == []


# --- Compose.from_config: Idefics3 -------------------------------------------


def test_idefics3_config_splits_large_images_into_normalized_patches():
    config = {
        "image_processor_type": "Idefics3ImageProcessor",
        "do_resize": True,
        "size": {"longest_edge": 8},
        "resample": 1,
        "do_image_splitting": True,
        "max_image_size": {"longest_edge": 4},
        "do_normalize": True,
        "image_mean": HALF,
        "image_std": HALF,
    }

    pipeline = Compose.from_config(config)
    (frames,) = pipeline([Image.new("RGB", (6, 3), (255, 0, 0))])

    assert _types(pipeline) == [
        ConvertToRGB,
        ResizeLongestEdge,
        PILtoNDarray,
        ResizeForVisionEncoder,
        ImageSplitter,
        Rescale,
        Normalize,
    ]
    assert _step(pipeline, 1, ResizeLongestEdge).resample is Image.Resampling.LANCZOS
    assert _step(pipeline, 1, ResizeLongestEdge).size == 8
    assert len(frames) == 3
    for frame in frames:
        assert frame.shape == (3, 4, 4)
        np.testing.assert_allclose(frame[:, 0, 0], [1.0, -1.0, -1.0], atol=1e-6)


def test_idefics3_without_splitting_square_resizes_to_the_default_max_size():
    pipeline = Compose.from_config({"image_processor_type": "Idefics3ImageProcessor"})

    assert _types(pipeline) == [ConvertToRGB, PILtoNDarray, SquareResize, Rescale]
    assert _step(pipeline, 2, SquareResize).size == 512
    assert _step(pipeline, 2, SquareResize).resample is Image.Resampling.LANCZOS


def test_idefics3_passes_a_missing_resample_through_to_pillow_defaults():
    config = {
        "image_processor_type": "Idefics3ImageProcessor",
        "do_resize": True,
        "size": {"longest_edge": 4},
        "resample": None,
        "max_image_size": {"longest_edge": 2},
    }

    pipeline = Compose.from_config(config)
    ((out,),) = pipeline([Image.new("RGB", (8, 8), (0, 0, 255))])

    assert _step(pipeline, 1, ResizeLongestEdge).resample is None
    assert _step(pipeline, 3, SquareResize).resample is None
    assert out.shape == (3, 2, 2)
    np.testing.assert_allclose(out[:, 0, 0], [0.0, 0.0, 1.0])


def test_idefics3_rejects_a_resize_without_longest_edge():
    config = {
        "image_processor_type": "Idefics3ImageProcessor",
        "do_resize": True,
        "size": {"shortest_edge": 4},
    }

    with pytest.raises(ValueError, match="must contain 'longest_edge' key for Idefics3"):
        Compose.from_config(config)


# --- interpolation names -----------------------------------------------------


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("LANCZOS", Image.Resampling.LANCZOS),
        ("Bilinear", Image.Resampling.BILINEAR),
        ("bicubic", Image.Resampling.BICUBIC),
        ("box", Image.Resampling.BOX),
        ("hamming", Image.Resampling.HAMMING),
    ],
)
def test_interpolation_resolver_maps_names_case_insensitively(name, expected):
    assert Compose._interpolation_resolver(name) is expected


def test_interpolation_resolver_rejects_a_missing_name():
    with pytest.raises(ValueError, match="Unknown interpolation method: None"):
        Compose._interpolation_resolver(None)
