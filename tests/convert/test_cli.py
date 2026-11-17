import argparse
from typing import cast

import pytest

from fastretrieval.convert.cli import build_parser, main


def test_parser_has_all_subcommands():
    parser = build_parser()
    actions = [a for a in parser._actions if a.dest == "command"]
    assert actions, "no subparsers registered"
    choices = cast(dict[str, argparse.ArgumentParser], actions[0].choices)
    assert set(choices) == {"onnx", "gguf", "verify", "card"}


def test_onnx_command_requires_source_and_out():
    parser = build_parser()
    args = parser.parse_args(["onnx", "acme/tiny-model", "--out", "/tmp/out"])
    assert args.command == "onnx"
    assert args.source == "acme/tiny-model"
    assert args.out == "/tmp/out"
    assert args.modality == "text"
    assert args.pooling is None
    assert args.normalize is None


def test_parser_accepts_explicit_non_qwen_contract_metadata():
    parser = build_parser()
    args = parser.parse_args(
        [
            "onnx",
            "acme/tiny-e5",
            "--out",
            "/tmp/out",
            "--modality",
            "text",
            "--pooling",
            "mean",
            "--normalize",
        ]
    )
    assert (args.modality, args.pooling, args.normalize) == ("text", "mean", True)


def test_conversion_commands_expose_the_optional_remote_backend():
    parser = build_parser()
    for command in ("onnx", "gguf"):
        args = parser.parse_args([command, "acme/tiny-model", "--out", "/tmp/out"])
        assert args.backend is None


def test_parser_has_no_flag_without_an_implementation():
    """Cờ khai trong parser mà không có nhánh xử lý = no-op im lặng."""
    parser = build_parser()
    choices = cast(
        dict[str, argparse.ArgumentParser],
        [a for a in parser._actions if a.dest == "command"][0].choices,
    )
    onnx = choices["onnx"]
    flags = {opt for a in onnx._actions for opt in a.option_strings}
    assert "--push" not in flags, "uploading is backlog; do not ship the flag before the behaviour"


def test_no_command_exits_nonzero(capsys):
    assert main([]) != 0


def test_parser_accepts_output_dim_for_cross_encoder_exports():
    parser = build_parser()
    args = parser.parse_args(
        [
            "onnx",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "--out",
            "/tmp/out",
            "--task",
            "text-classification",
            "--pooling",
            "cls",
            "--no-normalize",
            "--output-dim",
            "1",
        ]
    )
    assert args.output_dim == 1
    assert args.yes_no_head is False


def test_output_dim_defaults_to_none():
    parser = build_parser()
    args = parser.parse_args(["onnx", "acme/tiny-model", "--out", "/tmp/out"])
    assert args.output_dim is None


def test_gguf_command_has_no_output_dim_flag():
    """output-dim chỉ có ngữ nghĩa với export ONNX classification; gguf không nhận."""
    parser = build_parser()
    choices = cast(
        dict[str, argparse.ArgumentParser],
        [a for a in parser._actions if a.dest == "command"][0].choices,
    )
    gguf = choices["gguf"]
    flags = {opt for a in gguf._actions for opt in a.option_strings}
    assert "--output-dim" not in flags


def test_onnx_command_passes_output_dim_to_converter(monkeypatch):
    captured: dict[str, object] = {}

    def fake_convert_onnx(source: str, out_dir: str, **kwargs: object) -> dict[str, float]:
        captured.update(kwargs)
        return {}

    monkeypatch.setattr(
        "fastretrieval.convert.modal_backend.resolve_backend", lambda backend: "local"
    )
    monkeypatch.setattr("fastretrieval.convert.onnx.convert_onnx", fake_convert_onnx)

    exit_code = main(
        [
            "onnx",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "--out",
            "/tmp/out",
            "--task",
            "text-classification",
            "--pooling",
            "cls",
            "--no-normalize",
            "--output-dim",
            "1",
        ]
    )

    assert exit_code == 0
    assert captured["output_dim"] == 1
    assert captured["yes_no"] is None
    assert captured["task"] == "text-classification"


def test_onnx_remote_route_receives_output_dim(monkeypatch):
    captured: dict[str, object] = {}

    def fake_run_remote(command: str, **kwargs: object) -> dict[str, object]:
        captured["command"] = command
        captured.update(kwargs)
        return {"backend": "modal"}

    monkeypatch.setattr(
        "fastretrieval.convert.modal_backend.resolve_backend", lambda backend: "modal"
    )
    monkeypatch.setattr("fastretrieval.convert.modal_backend.run_remote", fake_run_remote)

    exit_code = main(
        [
            "onnx",
            "cross-encoder/ms-marco-MiniLM-L-6-v2",
            "--out",
            "/tmp/out",
            "--task",
            "text-classification",
            "--pooling",
            "cls",
            "--no-normalize",
            "--output-dim",
            "1",
        ]
    )

    assert exit_code == 0
    assert captured["command"] == "onnx"
    assert captured["output_dim"] == 1
    assert captured["yes_no"] is None


def test_missing_deps_message_names_the_requirements_file():
    from fastretrieval.convert import require_convert_deps

    with pytest.raises(ImportError, match="requirements.txt"):
        require_convert_deps("a_module_that_does_not_exist_anywhere")


def _converted_dir(tmp_path, *, task, formats, artifacts):
    from fastretrieval.contract import ModelContract, PreprocessorSpec
    from fastretrieval.convert.manifest import write_manifest

    directory = tmp_path / "converted"
    contract = ModelContract(
        model_id="acme/tiny-model",
        source="acme/tiny-model",
        task=task,
        modality="text",
        model_family="qwen3",
        output_dim=2 if task == "generative_reranker" else 4,
        output_shape=(2,) if task == "generative_reranker" else (4,),
        pooling="LAST_TOKEN",
        normalization=task == "dense",
        max_seq_len=32,
        preprocessor=PreprocessorSpec(kind="text"),
        artifact_formats=formats,
        tokenizer_files=("tokenizer.json",),
        quantization="Q4_K_M" if formats == ("gguf",) else "int8,q4f16",
    )
    (directory / "tokenizer.json").parent.mkdir(parents=True)
    (directory / "tokenizer.json").write_text("{}", encoding="utf-8")
    for relative, megabytes in artifacts.items():
        path = directory / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"\0" * (megabytes * 1024 * 1024))
    write_manifest(directory, contract)
    return directory


def test_card_command_writes_an_onnx_embedding_card_with_real_sizes(tmp_path, capsys):
    directory = _converted_dir(
        tmp_path,
        task="dense",
        formats=("onnx",),
        artifacts={"onnx/model_quantized.onnx": 3, "onnx/model_q4f16.onnx": 2},
    )

    exit_code = main(["card", str(directory), "--source", "acme/tiny-model"])

    readme = directory / "README.md"
    assert exit_code == 0
    assert capsys.readouterr().out.strip() == str(readme)
    card = readme.read_text(encoding="utf-8")
    assert "# ONNX build of acme/tiny-model" in card
    assert "pipeline_tag: feature-extraction" in card
    assert "| int8 | `onnx/model_quantized.onnx` | 3 MB |" in card
    assert "| q4f16 | `onnx/model_q4f16.onnx` | 2 MB |" in card


def test_card_command_lists_nested_gguf_files_for_a_reranker(tmp_path, capsys):
    directory = _converted_dir(
        tmp_path,
        task="generative_reranker",
        formats=("gguf",),
        artifacts={"variants/tiny-model-q4-k-m.GGUF": 1},
    )

    exit_code = main(["card", str(directory), "--source", "acme/tiny-model"])

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == str(directory / "README.md")
    card = (directory / "README.md").read_text(encoding="utf-8")
    assert "# GGUF build of acme/tiny-model" in card
    assert "pipeline_tag: text-classification" in card
    assert "from fastretrieval import TextCrossEncoder" in card
    assert "| variants/tiny-model-q4-k-m.GGUF | `variants/tiny-model-q4-k-m.GGUF` | 1 MB |" in card


def test_card_command_refuses_a_manifest_from_another_source(tmp_path):
    directory = _converted_dir(
        tmp_path, task="dense", formats=("onnx",), artifacts={"onnx/model_quantized.onnx": 1}
    )

    with pytest.raises(ValueError, match="manifest source mismatch"):
        main(["card", str(directory), "--source", "acme/other-model"])

    assert not (directory / "README.md").exists()
