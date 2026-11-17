import json
from pathlib import Path
from typing import Any, cast

import pytest

from fastretrieval.contract import PreprocessorSpec


def test_default_is_text():
    spec = PreprocessorSpec()
    assert spec.kind == "text"
    assert spec.config_file is None


def test_from_model_dir_returns_none_when_no_config(tmp_path: Path):
    assert PreprocessorSpec.from_model_dir(tmp_path) is None


def test_from_model_dir_reads_image_config(tmp_path: Path):
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "image_processor_type": "CLIPImageProcessor",
                "size": {"height": 224, "width": 224},
                "image_mean": [0.481, 0.457, 0.408],
                "image_std": [0.268, 0.261, 0.275],
            }
        ),
        encoding="utf-8",
    )
    spec = PreprocessorSpec.from_model_dir(tmp_path)
    assert spec is not None
    assert spec.kind == "image"
    assert spec.image_size == (224, 224)
    assert spec.image_mean == (0.481, 0.457, 0.408)
    assert spec.image_std == (0.268, 0.261, 0.275)


def test_from_model_dir_reads_shorthand_size(tmp_path: Path):
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps({"image_processor_type": "ViTImageProcessor", "size": 256}),
        encoding="utf-8",
    )
    spec = PreprocessorSpec.from_model_dir(tmp_path)
    assert spec is not None
    assert spec.image_size == (256, 256)


def test_tokenizer_only_config_is_text(tmp_path: Path):
    (tmp_path / "preprocessor_config.json").write_text(
        json.dumps({"tokenizer_class": "Qwen2Tokenizer"}), encoding="utf-8"
    )
    spec = PreprocessorSpec.from_model_dir(tmp_path)
    assert spec is not None
    assert spec.kind == "text"


def test_malformed_config_raises_with_path_in_message(tmp_path: Path):
    (tmp_path / "preprocessor_config.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="preprocessor_config.json"):
        PreprocessorSpec.from_model_dir(tmp_path)


def test_custom_model_spec_defaults_to_no_preprocessor():
    from fastretrieval import CustomModelSpec

    spec = CustomModelSpec(model_id="acme/tiny", hf="acme/tiny", dim=384)
    assert spec.preprocessor is None
    assert spec.pooling.value == "MEAN"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"kind": "pixels"}, "unsupported preprocessor kind: 'pixels'"),
        ({"kind": "image", "image_size": (224,)}, "image_size must contain two positive values"),
        ({"kind": "image", "image_size": (0, 224)}, "image_size must contain two positive values"),
        ({"kind": "image", "image_mean": (0.5, 0.5)}, "image_mean must contain three channel"),
        ({"kind": "image", "image_std": (0.5, 0.5)}, "image_std must contain three non-zero"),
        ({"kind": "image", "image_std": (0.5, 0.0, 0.5)}, "image_std must contain three non-zero"),
    ],
)
def test_spec_rejects_invalid_values(kwargs: dict, message: str):
    with pytest.raises(ValueError, match=message):
        PreprocessorSpec(**kwargs)


def test_from_dict_rejects_non_object():
    with pytest.raises(ValueError, match="preprocessor must be an object, got str"):
        # The non-dict argument is deliberate: it exercises the runtime type check.
        PreprocessorSpec.from_dict(cast(dict[str, Any], "text"))


def test_from_dict_rejects_missing_kind():
    with pytest.raises(ValueError, match="preprocessor is missing kind"):
        PreprocessorSpec.from_dict({"config_file": "preprocessor_config.json"})


def test_from_dict_coerces_image_fields_and_roundtrips_through_to_dict():
    spec = PreprocessorSpec.from_dict(
        {
            "kind": "image",
            "config_file": "preprocessor_config.json",
            "image_size": ["224", 224.0],
            "image_mean": [0, 1, "0.5"],
            "image_std": [1, 1, 1],
        }
    )

    assert spec == PreprocessorSpec(
        kind="image",
        config_file="preprocessor_config.json",
        image_size=(224, 224),
        image_mean=(0.0, 1.0, 0.5),
        image_std=(1.0, 1.0, 1.0),
    )
    assert spec.to_dict() == {
        "config_file": "preprocessor_config.json",
        "image_mean": [0.0, 1.0, 0.5],
        "image_size": [224, 224],
        "image_std": [1.0, 1.0, 1.0],
        "kind": "image",
    }
    assert PreprocessorSpec.from_dict(spec.to_dict()) == spec


def test_text_spec_serializes_absent_image_fields_as_null():
    assert PreprocessorSpec.from_dict({"kind": "text"}).to_dict() == {
        "config_file": None,
        "image_mean": None,
        "image_size": None,
        "image_std": None,
        "kind": "text",
    }


def _write_config(tmp_path: Path, config: dict) -> Path:
    (tmp_path / "preprocessor_config.json").write_text(json.dumps(config), encoding="utf-8")
    return tmp_path


def test_image_mean_alone_marks_config_as_image_without_size(tmp_path: Path):
    spec = PreprocessorSpec.from_model_dir(
        _write_config(tmp_path, {"image_mean": [0.5, 0.5, 0.5]})
    )

    assert spec == PreprocessorSpec(
        kind="image",
        config_file="preprocessor_config.json",
        image_size=None,
        image_mean=(0.5, 0.5, 0.5),
        image_std=None,
    )


def test_shortest_edge_size_becomes_square(tmp_path: Path):
    spec = PreprocessorSpec.from_model_dir(
        _write_config(
            tmp_path,
            {"image_processor_type": "CLIPImageProcessor", "size": {"shortest_edge": 336}},
        )
    )

    assert spec is not None
    assert spec.image_size == (336, 336)


@pytest.mark.parametrize("size", [True, "224", {"longest_edge": 224}, {"height": 224}])
def test_unsupported_size_field_is_rejected(tmp_path: Path, size: object):
    _write_config(tmp_path, {"image_processor_type": "ViTImageProcessor", "size": size})

    with pytest.raises(ValueError, match="unsupported preprocessor size field"):
        PreprocessorSpec.from_model_dir(tmp_path)


@pytest.mark.parametrize("mean", [0.5, ["a", "b", "c"]])
def test_non_numeric_channel_values_are_rejected(tmp_path: Path, mean: object):
    _write_config(tmp_path, {"image_processor_type": "ViTImageProcessor", "image_mean": mean})

    with pytest.raises(ValueError, match="expected three channel values, got"):
        PreprocessorSpec.from_model_dir(tmp_path)


def test_wrong_channel_count_is_rejected(tmp_path: Path):
    _write_config(tmp_path, {"image_processor_type": "ViTImageProcessor", "image_std": [0.2, 0.2]})

    with pytest.raises(ValueError, match="expected 3 channel values, got 2"):
        PreprocessorSpec.from_model_dir(tmp_path)
