"""Support-matrix resolution and contract building fail closed on every metadata gap."""

import dataclasses
import json
import sys
from pathlib import Path
from typing import cast

import pytest

from fastretrieval.common.model_description import PoolingType
from fastretrieval.contract import PreprocessorSpec
from fastretrieval.convert.profiles import resolve_profile


def _model_dir(tmp_path: Path, config: object, name: str = "model") -> Path:
    directory = tmp_path / name
    directory.mkdir()
    payload = config if isinstance(config, str) else json.dumps(config)
    (directory / "config.json").write_text(payload, encoding="utf-8")
    return directory


def _bert_profile(tmp_path: Path, task: str = "dense", **config: object):
    source = _model_dir(tmp_path, {"model_type": "bert", "hidden_size": 8, **config})
    return resolve_profile(source, task=task, modality="text")


def _qwen3_reranker_profile(tmp_path: Path):
    source = _model_dir(tmp_path, {"model_type": "qwen3", "hidden_size": 8})
    return resolve_profile(source, task="generative-reranker", modality="text")


def _serve_config_from_hub(monkeypatch, config_path: Path, calls: list) -> None:
    # Boundary (c): huggingface_hub would download config.json over the network.
    def hf_hub_download(*, repo_id: str, filename: str) -> str:
        calls.append((repo_id, filename))
        return str(config_path)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", hf_hub_download)


# ---------------------------------------------------------------------------
# build_contract
# ---------------------------------------------------------------------------
def test_build_contract_rejects_non_boolean_normalization(tmp_path):
    profile = _bert_profile(tmp_path)

    with pytest.raises(ValueError, match="normalization must be boolean"):
        profile.build_contract(pooling="mean", normalization="yes")


def test_build_contract_rejects_pooling_outside_the_profile_modes(tmp_path):
    profile = _bert_profile(tmp_path)

    with pytest.raises(ValueError, match="pooling 'DISABLED' is not supported"):
        profile.build_contract(pooling="disabled", normalization=True)


def test_build_contract_rejects_an_unknown_pooling_name(tmp_path):
    profile = _bert_profile(tmp_path)

    with pytest.raises(ValueError, match="unsupported pooling 'max'"):
        profile.build_contract(pooling="max", normalization=True)


def test_build_contract_accepts_a_pooling_enum_member(tmp_path):
    contract = _bert_profile(tmp_path).build_contract(pooling=PoolingType.CLS, normalization=False)

    assert contract.pooling is PoolingType.CLS
    assert contract.output_dim == 8


def test_build_contract_rejects_normalization_outside_the_profile_modes(tmp_path):
    profile = dataclasses.replace(_bert_profile(tmp_path), normalization_modes=frozenset({False}))

    with pytest.raises(ValueError, match="normalization=True is not supported"):
        profile.build_contract(pooling="mean", normalization=True)


def test_build_contract_rejects_a_preprocessor_that_is_not_a_spec(tmp_path):
    profile = _bert_profile(tmp_path)

    with pytest.raises(ValueError, match="preprocessor must be a PreprocessorSpec"):
        profile.build_contract(pooling="mean", normalization=True, preprocessor={"kind": "text"})


def test_yes_no_head_declares_two_unnormalized_logits(tmp_path):
    profile = _qwen3_reranker_profile(tmp_path)

    contract = profile.build_contract(
        pooling="last_token", normalization=False, yes_no=("yes", "no")
    )

    assert (contract.task, contract.output_dim, contract.output_shape) == (
        "generative_reranker",
        2,
        (2,),
    )
    with pytest.raises(ValueError, match="yes_no logits cannot declare normalization=True"):
        profile.build_contract(pooling="last_token", normalization=True, yes_no=("yes", "no"))


def test_dense_profile_without_any_dimension_requires_explicit_output_metadata(tmp_path):
    source = _model_dir(tmp_path, {"model_type": "bert"})
    profile = resolve_profile(source, task="dense", modality="text")

    assert profile.output_dim is None
    with pytest.raises(ValueError, match="output_dim or output_shape metadata is required"):
        profile.build_contract(pooling="mean", normalization=True)


def test_single_axis_output_shape_supplies_the_output_dim(tmp_path):
    source = _model_dir(tmp_path, {"model_type": "bert"})
    profile = resolve_profile(source, task="dense", modality="text")

    contract = profile.build_contract(pooling="mean", normalization=True, output_shape=(6,))

    assert (contract.output_dim, contract.output_shape) == (6, (6,))


def test_multi_axis_output_shape_leaves_the_output_dim_unset(tmp_path):
    source = _model_dir(tmp_path, {"model_type": "bert"})
    profile = resolve_profile(source, task="dense", modality="text")

    contract = profile.build_contract(pooling="mean", normalization=True, output_shape=(4, 6))

    assert (contract.output_dim, contract.output_shape) == (None, (4, 6))


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"tokenizer_files": ()}, "tokenizer_files metadata is required"),
        ({"artifact_formats": ()}, "artifact_formats metadata is required"),
    ],
)
def test_build_contract_requires_tokenizer_files_and_artifact_formats(tmp_path, override, message):
    profile = _bert_profile(tmp_path)

    with pytest.raises(ValueError, match=message):
        profile.build_contract(pooling="mean", normalization=True, **override)


# ---------------------------------------------------------------------------
# resolve_profile
# ---------------------------------------------------------------------------
def test_resolve_profile_rejects_an_unknown_task(tmp_path):
    source = _model_dir(tmp_path, {"model_type": "bert", "hidden_size": 8})

    with pytest.raises(ValueError, match="task=summarization modality=text: unsupported task"):
        resolve_profile(source, task="summarization", modality="text")


def test_resolve_profile_rejects_an_unknown_modality(tmp_path):
    source = _model_dir(tmp_path, {"model_type": "bert", "hidden_size": 8})

    with pytest.raises(ValueError, match="modality=smell: unsupported modality"):
        resolve_profile(source, task="dense", modality="smell")


def test_bert_profile_rejects_the_generative_reranker_task(tmp_path):
    source = _model_dir(tmp_path, {"model_type": "bert", "hidden_size": 8})

    with pytest.raises(ValueError, match="task is not supported for 'bert'"):
        resolve_profile(source, task="generative-reranker", modality="text")


@pytest.mark.parametrize(
    ("config", "architecture", "family"),
    [
        (
            {"model_type": "custom-bert", "architectures": ["BertModel"], "hidden_size": 8},
            "bertmodel",
            "bert",
        ),
        ({"architectures": ["Qwen3Model"], "hidden_size": 8}, "qwen3model", "qwen3"),
        (
            {"model_type": "", "architectures": ["Qwen3Model"], "hidden_size": 8},
            "qwen3model",
            "qwen3",
        ),
    ],
)
def test_architecture_falls_back_to_the_declared_architectures_list(
    tmp_path, config, architecture, family
):
    profile = resolve_profile(_model_dir(tmp_path, config), task="dense", modality="text")

    assert (profile.architecture, profile.model_family) == (architecture, family)


@pytest.mark.parametrize("family", ["", "   ", 7])
def test_model_family_override_must_be_a_non_empty_string(tmp_path, family):
    source = _model_dir(
        tmp_path,
        {"model_type": "bert", "hidden_size": 8, "fastretrieval_model_family": family},
    )

    with pytest.raises(ValueError, match="fastretrieval_model_family must be a non-empty string"):
        resolve_profile(source, task="dense", modality="text")


@pytest.mark.parametrize(
    ("config", "output_dim"),
    [
        ({"sentence_embedding_dimension": 16, "projection_dim": 12, "hidden_size": 8}, 16),
        ({"projection_dim": 12, "hidden_size": 8}, 12),
        ({"hidden_size": 8}, 8),
        ({"sentence_embedding_dimension": True, "projection_dim": 0, "hidden_size": -3}, None),
    ],
)
def test_dense_output_dim_prefers_sentence_embedding_then_projection_then_hidden_size(
    tmp_path, config, output_dim
):
    source = _model_dir(tmp_path, {"model_type": "bert", **config})

    assert resolve_profile(source, task="dense", modality="text").output_dim == output_dim


def test_non_dense_tasks_do_not_inherit_hidden_size_as_output_dim(tmp_path):
    profile = _bert_profile(tmp_path, task="cross_encoder")

    assert profile.task == "cross_encoder"
    assert profile.output_dim is None


@pytest.mark.parametrize(
    ("config", "max_seq_len"),
    [
        ({"max_seq_len": 64, "max_position_embeddings": 512}, 64),
        ({"max_position_embeddings": 512}, 512),
        ({}, None),
    ],
)
def test_max_seq_len_prefers_the_explicit_key(tmp_path, config, max_seq_len):
    assert _bert_profile(tmp_path, **config).max_seq_len == max_seq_len


def test_local_preprocessor_config_replaces_the_default_spec_and_fails_closed(tmp_path):
    source = _model_dir(tmp_path, {"model_type": "bert", "hidden_size": 8})
    (source / "preprocessor_config.json").write_text(
        json.dumps(
            {
                "image_processor_type": "ViTImageProcessor",
                "image_mean": [0.5, 0.5, 0.5],
                "image_std": [0.25, 0.25, 0.25],
            }
        ),
        encoding="utf-8",
    )

    profile = resolve_profile(source, task="dense", modality="text")

    assert profile.preprocessor.kind == "image"
    assert profile.preprocessor.image_mean == (0.5, 0.5, 0.5)
    with pytest.raises(ValueError, match="preprocessor kind 'image' does not match"):
        profile.build_contract(pooling="mean", normalization=True)


def test_config_json_file_path_resolves_its_parent_directory(tmp_path):
    source = _model_dir(tmp_path, {"model_type": "bert", "hidden_size": 8})
    config_path = source / "config.json"

    profile = resolve_profile(config_path, task="dense", modality="text")

    assert profile.source == str(config_path)
    assert (profile.architecture, profile.output_dim) == ("bert", 8)


@pytest.mark.parametrize(
    ("names", "message"),
    [
        ([], "input_names must be a sequence of non-empty strings"),
        (["input_ids", " "], "input_names must be a sequence of non-empty strings"),
        (["input_ids", 3], "input_names must be a sequence of non-empty strings"),
        (["input_ids", "input_ids"], "input_names must not contain duplicates"),
    ],
)
def test_invalid_input_names_fail_closed(tmp_path, names, message):
    source = _model_dir(tmp_path, {"model_type": "bert", "hidden_size": 8, "input_names": names})

    with pytest.raises(ValueError, match=message):
        resolve_profile(source, task="dense", modality="text")


def test_declared_input_names_are_kept_in_order(tmp_path):
    names = ["input_ids", "attention_mask", "token_type_ids"]

    assert _bert_profile(tmp_path, input_names=names).input_names == tuple(names)


# ---------------------------------------------------------------------------
# config.json loading
# ---------------------------------------------------------------------------
def test_local_directory_without_config_json_is_rejected(tmp_path):
    source = tmp_path / "empty-model"
    source.mkdir()

    with pytest.raises(ValueError, match="config.json is required for profile resolution"):
        resolve_profile(source, task="dense", modality="text")


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ("{not json", "cannot read config.json"),
        ("[1, 2]", "config.json must be a JSON object"),
    ],
)
def test_unreadable_local_config_json_is_rejected(tmp_path, payload, message):
    source = _model_dir(tmp_path, payload)

    with pytest.raises(ValueError, match=message):
        resolve_profile(source, task="dense", modality="text")


def test_hub_profile_reads_config_json_and_keeps_the_default_preprocessor(tmp_path, monkeypatch):
    config_path = tmp_path / "hub-config.json"
    config_path.write_text(json.dumps({"model_type": "bert", "hidden_size": 8}), encoding="utf-8")
    calls: list = []
    _serve_config_from_hub(monkeypatch, config_path, calls)

    profile = resolve_profile("acme/tiny-bert", task="dense", modality="text")

    assert calls == [("acme/tiny-bert", "config.json")]
    assert profile.config["_model_id"] == "acme/tiny-bert"
    assert profile.preprocessor == PreprocessorSpec(
        kind="text", config_file="preprocessor_config.json"
    )


def test_hub_profile_without_huggingface_hub_asks_for_a_local_directory(monkeypatch):
    # Boundary (c): a None entry makes ``from huggingface_hub import ...`` raise ImportError,
    # which is what an install without huggingface-hub does.
    monkeypatch.setitem(sys.modules, "huggingface_hub", None)

    with pytest.raises(ValueError, match="install huggingface-hub or provide a local model"):
        resolve_profile("acme/tiny-bert", task="dense", modality="text")


def test_hub_download_failure_is_reported_with_the_source(monkeypatch):
    # Boundary (c): the hub download is the network call that fails here.
    def hf_hub_download(*, repo_id: str, filename: str) -> str:
        raise OSError("hub unreachable")

    monkeypatch.setattr("huggingface_hub.hf_hub_download", hf_hub_download)

    with pytest.raises(
        ValueError, match=r"source='acme/tiny-bert': cannot inspect config.json.*hub unreachable"
    ):
        resolve_profile("acme/tiny-bert", task="dense", modality="text")


def test_hub_config_json_must_be_an_object(tmp_path, monkeypatch):
    config_path = tmp_path / "hub-config.json"
    config_path.write_text("[]", encoding="utf-8")
    _serve_config_from_hub(monkeypatch, config_path, [])

    with pytest.raises(ValueError, match="config.json must be a JSON object"):
        resolve_profile("acme/tiny-bert", task="dense", modality="text")


class _UnresolvablePathLike:
    """A path-like whose filesystem path cannot be produced."""

    def __fspath__(self) -> str:
        raise ValueError("path is not representable")

    def __str__(self) -> str:
        return "acme/path-like-model"


def test_unconvertible_path_like_source_is_resolved_as_a_hub_id(tmp_path, monkeypatch):
    config_path = tmp_path / "hub-config.json"
    config_path.write_text(json.dumps({"model_type": "bert", "hidden_size": 8}), encoding="utf-8")
    calls: list = []
    _serve_config_from_hub(monkeypatch, config_path, calls)

    # resolve_profile types source as str | Path; a bare os.PathLike reaches its hub fallback.
    source = cast(Path, _UnresolvablePathLike())
    profile = resolve_profile(source, task="dense", modality="text")

    assert calls == [("acme/path-like-model", "config.json")]
    assert profile.source == "acme/path-like-model"
