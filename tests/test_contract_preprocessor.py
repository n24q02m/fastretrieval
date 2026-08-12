import json
from pathlib import Path

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
