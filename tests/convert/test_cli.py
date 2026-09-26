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


def test_missing_deps_message_names_the_requirements_file():
    from fastretrieval.convert import require_convert_deps

    with pytest.raises(ImportError, match="requirements.txt"):
        require_convert_deps("a_module_that_does_not_exist_anywhere")


def test_onnx_parser_accepts_output_dim_for_cross_encoder():
    parser = build_parser()
    args = parser.parse_args(
        [
            "onnx",
            "acme/tiny-cross-encoder",
            "--out",
            "/tmp/out",
            "--task",
            "text-classification",
            "--output-dim",
            "3",
        ]
    )
    assert args.task == "text-classification"
    assert args.output_dim == 3


def test_output_dim_defaults_to_none():
    parser = build_parser()
    args = parser.parse_args(["onnx", "acme/tiny-model", "--out", "/tmp/out"])
    assert args.output_dim is None


def test_main_threads_output_dim_through_local_conversion(monkeypatch):
    captured: dict = {}

    def fake_convert(source, out, **kwargs):
        captured.update(kwargs)
        return {"int8": 1.0}

    monkeypatch.setattr(
        "fastretrieval.convert.modal_backend.resolve_backend", lambda name: "local"
    )
    monkeypatch.setattr("fastretrieval.convert.onnx.convert_onnx", fake_convert)

    exit_code = main(
        [
            "onnx",
            "acme/tiny-cross-encoder",
            "--out",
            "/tmp/out",
            "--task",
            "text-classification",
            "--output-dim",
            "3",
        ]
    )

    assert exit_code == 0
    assert captured["output_dim"] == 3
    assert captured["task"] == "text-classification"


def test_main_threads_output_dim_through_remote_conversion(monkeypatch):
    captured: dict = {}

    def fake_run_remote(command, **kwargs):
        captured.update(kwargs)
        return {"backend": "modal", "command": command}

    monkeypatch.setattr(
        "fastretrieval.convert.modal_backend.resolve_backend", lambda name: "modal"
    )
    monkeypatch.setattr("fastretrieval.convert.modal_backend.run_remote", fake_run_remote)

    exit_code = main(["onnx", "acme/tiny-cross-encoder", "--out", "/tmp/out", "--output-dim", "1"])

    assert exit_code == 0
    assert captured["output_dim"] == 1
