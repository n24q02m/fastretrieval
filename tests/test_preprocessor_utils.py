import json
from pathlib import Path

import pytest
from tokenizers import Tokenizer, models

from qwen3_embed.common.preprocessor_utils import load_special_tokens, load_tokenizer


@pytest.fixture
def create_test_files(tmp_path):
    """
    Helper fixture to create necessary tokenizer files in a temporary directory.
    """

    def _create_files(
        config=None, tokenizer_config=None, special_tokens_map=None, files_to_skip=None
    ):
        if files_to_skip is None:
            files_to_skip = set()

        # Default config
        final_config = {"pad_token_id": 0}
        if config is not None:
            final_config = config

        # Default tokenizer config
        final_tokenizer_config = {"model_max_length": 512, "pad_token": "<pad>"}
        if tokenizer_config is not None:
            final_tokenizer_config = tokenizer_config

        # Create dummy tokenizer using BPE model
        tokenizer = Tokenizer(models.BPE())
        # Add special tokens so they exist in the vocab
        tokenizer.add_special_tokens(["<pad>", "<unk>", "[CLS]", "[SEP]"])

        # Write files
        if "config.json" not in files_to_skip:
            with open(tmp_path / "config.json", "w") as f:
                json.dump(final_config, f)

        if "tokenizer.json" not in files_to_skip:
            tokenizer.save(str(tmp_path / "tokenizer.json"))

        if "tokenizer_config.json" not in files_to_skip:
            with open(tmp_path / "tokenizer_config.json", "w") as f:
                json.dump(final_tokenizer_config, f)

        if special_tokens_map and "special_tokens_map.json" not in files_to_skip:
            with open(tmp_path / "special_tokens_map.json", "w") as f:
                json.dump(special_tokens_map, f)

        return tmp_path

    return _create_files


def test_load_tokenizer_success(create_test_files):
    model_dir = create_test_files()
    tokenizer, special_tokens = load_tokenizer(model_dir)

    assert isinstance(tokenizer, Tokenizer)
    assert isinstance(special_tokens, dict)

    # Check truncation enabled (default max_length 512)
    # The truncation params are not directly accessible as simple properties easily,
    # but we can check usage or if it throws.
    # Actually tokenizer.truncation returns a dict if set.
    assert tokenizer.truncation is not None
    assert tokenizer.truncation["max_length"] == 512

    # Check padding enabled
    assert tokenizer.padding is not None
    assert tokenizer.padding["pad_id"] == 0
    assert tokenizer.padding["pad_token"] == "<pad>"


def test_load_tokenizer_missing_config(create_test_files):
    model_dir = create_test_files(files_to_skip={"config.json"})
    with pytest.raises(ValueError, match="Could not find config.json"):
        load_tokenizer(model_dir)


def test_load_tokenizer_missing_tokenizer_json(create_test_files):
    model_dir = create_test_files(files_to_skip={"tokenizer.json"})
    with pytest.raises(ValueError, match="Could not find tokenizer.json"):
        load_tokenizer(model_dir)


def test_load_tokenizer_missing_tokenizer_config(create_test_files):
    model_dir = create_test_files(files_to_skip={"tokenizer_config.json"})
    with pytest.raises(ValueError, match="Could not find tokenizer_config.json"):
        load_tokenizer(model_dir)


def test_load_tokenizer_invalid_max_length(create_test_files):
    # Pass a tokenizer_config without max_length or model_max_length
    invalid_config = {"pad_token": "<pad>"}
    model_dir = create_test_files(tokenizer_config=invalid_config)

    with pytest.raises(
        ValueError, match="Models without model_max_length or max_length are not supported."
    ):
        load_tokenizer(model_dir)


def test_load_tokenizer_with_special_tokens(create_test_files):
    special_map = {"cls_token": "[CLS]", "sep_token": {"content": "[SEP]", "lstrip": False}}
    model_dir = create_test_files(special_tokens_map=special_map)

    tokenizer, special_tokens = load_tokenizer(model_dir)

    # Verify tokens are added to the tokenizer
    # Since we added them in setup, token_to_id should return an ID.
    assert tokenizer.token_to_id("[CLS]") is not None
    assert tokenizer.token_to_id("[SEP]") is not None

    # Verify the returned map
    assert "[CLS]" in special_tokens
    assert "[SEP]" in special_tokens
    assert special_tokens["[CLS]"] == tokenizer.token_to_id("[CLS]")


def test_load_tokenizer_padding_logic(create_test_files):
    # Test with custom pad token and ID
    config = {"pad_token_id": 99}
    tokenizer_config = {
        "model_max_length": 128,
        "pad_token": "<unk>",
    }  # Using <unk> as pad for test

    model_dir = create_test_files(config=config, tokenizer_config=tokenizer_config)
    tokenizer, _ = load_tokenizer(model_dir)

    assert tokenizer.padding["pad_id"] == 99
    assert tokenizer.padding["pad_token"] == "<unk>"
    assert tokenizer.truncation["max_length"] == 128


def test_load_tokenizer_max_length_precedence(create_test_files):
    # Test precedence: max_length vs model_max_length (min is taken)
    tokenizer_config = {"model_max_length": 100, "max_length": 200, "pad_token": "<pad>"}
    model_dir = create_test_files(tokenizer_config=tokenizer_config)
    tokenizer, _ = load_tokenizer(model_dir)
    assert tokenizer.truncation["max_length"] == 100

    tokenizer_config = {"model_max_length": 300, "max_length": 150, "pad_token": "<pad>"}
    model_dir = create_test_files(tokenizer_config=tokenizer_config)
    tokenizer, _ = load_tokenizer(model_dir)
    assert tokenizer.truncation["max_length"] == 150


class TestLoadSpecialTokens:
    """Tests for load_special_tokens utility."""

    def test_file_exists_and_valid(self, tmp_path: Path) -> None:
        """Test loading special tokens when the file exists and is valid JSON."""
        model_dir = tmp_path / "model_dir"
        model_dir.mkdir()
        tokens_map_path = model_dir / "special_tokens_map.json"

        expected_tokens = {"unk_token": "<unk>", "pad_token": "<pad>"}
        with open(tokens_map_path, "w") as f:
            json.dump(expected_tokens, f)

        result = load_special_tokens(model_dir)
        assert result == expected_tokens

    def test_file_missing(self, tmp_path: Path) -> None:
        """Test loading special tokens when the file does not exist."""
        model_dir = tmp_path / "model_dir"
        model_dir.mkdir()

        result = load_special_tokens(model_dir)
        assert result == {}

    def test_file_invalid_json(self, tmp_path: Path) -> None:
        """Test loading special tokens when the file contains invalid JSON."""
        model_dir = tmp_path / "model_dir"
        model_dir.mkdir()
        tokens_map_path = model_dir / "special_tokens_map.json"

        with open(tokens_map_path, "w") as f:
            f.write("invalid json")

        with pytest.raises(json.JSONDecodeError):
            load_special_tokens(model_dir)
