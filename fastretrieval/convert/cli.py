"""Command-line interface for the optional model converter."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m fastretrieval.convert",
        description="Convert a HuggingFace model into a fastretrieval-ready artifact.",
    )
    sub = parser.add_subparsers(dest="command")

    onnx = sub.add_parser("onnx", help="export to ONNX, then quantize to INT8 and Q4F16")
    onnx.add_argument("source", help="HuggingFace model id or a local directory")
    onnx.add_argument("--out", required=True, help="output directory")
    onnx.add_argument(
        "--task",
        default="feature-extraction",
        help="optimum export task (default: feature-extraction)",
    )
    onnx.add_argument(
        "--modality",
        choices=("text", "image", "audio", "video"),
        default="text",
        help="input modality declared in the model contract",
    )
    onnx.add_argument(
        "--pooling",
        choices=("cls", "mean", "last_token", "disabled"),
        default=None,
        help="explicit pooling; required when the profile cannot infer it",
    )
    onnx.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="declare output L2 normalization instead of guessing",
    )
    onnx.add_argument(
        "--variants",
        default="int8,q4f16",
        help="comma-separated subset of int8,q4f16 (default: both)",
    )
    onnx.add_argument(
        "--yes-no-head",
        action="store_true",
        help="reduce a causal-LM reranker to two logits (see --yes-token/--no-token)",
    )
    onnx.add_argument("--yes-token", default="yes", help="token meaning relevant")
    onnx.add_argument("--no-token", default="no", help="token meaning not relevant")

    gguf = sub.add_parser("gguf", help="convert to GGUF and quantize")
    gguf.add_argument("source", help="HuggingFace model id or a local directory")
    gguf.add_argument("--out", required=True, help="output directory")
    gguf.add_argument(
        "--task",
        default="feature-extraction",
        help="model task declared in the contract (default: feature-extraction)",
    )
    gguf.add_argument(
        "--modality",
        choices=("text", "image", "audio", "video"),
        default="text",
        help="input modality declared in the model contract",
    )
    gguf.add_argument(
        "--pooling",
        choices=("cls", "mean", "last_token", "disabled"),
        default=None,
        help="explicit pooling; required when the profile cannot infer it",
    )
    gguf.add_argument(
        "--normalize",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="declare output L2 normalization instead of guessing",
    )
    gguf.add_argument("--quant", default="Q4_K_M", help="llama-quantize type (default: Q4_K_M)")
    gguf.add_argument(
        "--llama-cpp",
        default=None,
        help="path to a llama.cpp checkout (default: $LLAMA_CPP_HOME)",
    )
    for conversion_parser in (onnx, gguf):
        conversion_parser.add_argument(
            "--backend",
            default=None,
            choices=("local", "modal"),
            help="where to run the conversion (default: local)",
        )

    verify = sub.add_parser("verify", help="compare a converted model against the torch reference")
    verify.add_argument("converted", help="directory holding the converted model")
    verify.add_argument("--source", required=True, help="original HuggingFace model id")
    verify.add_argument(
        "--atol", type=float, default=1e-2, help="absolute tolerance (default: 1e-2)"
    )

    card = sub.add_parser("card", help="write a model card for a converted directory")
    card.add_argument("directory", help="directory holding the converted model")
    card.add_argument("--source", required=True, help="original HuggingFace model id")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 2
    if args.command == "onnx":
        from fastretrieval.convert.modal_backend import resolve_backend, run_remote
        from fastretrieval.convert.onnx import convert_onnx

        yes_no = (args.yes_token, args.no_token) if args.yes_no_head else None
        if resolve_backend(args.backend) == "modal":
            result = run_remote(
                "onnx",
                source=args.source,
                out_dir=args.out,
                task=args.task,
                modality=args.modality,
                variants=[
                    variant.strip() for variant in args.variants.split(",") if variant.strip()
                ],
                pooling=args.pooling,
                normalization=args.normalize,
                yes_no=yes_no,
            )
            print(result)
            return 0
        sizes = convert_onnx(
            args.source,
            args.out,
            task=args.task,
            modality=args.modality,
            variants=[variant.strip() for variant in args.variants.split(",") if variant.strip()],
            pooling=args.pooling,
            normalization=args.normalize,
            yes_no=yes_no,
        )
        for name, megabytes in sizes.items():
            print(f"{name}: {megabytes:.1f} MB")
        return 0
    if args.command == "gguf":
        from fastretrieval.convert.gguf import convert_gguf
        from fastretrieval.convert.modal_backend import resolve_backend, run_remote

        if resolve_backend(args.backend) == "modal":
            result = run_remote(
                "gguf",
                source=args.source,
                out_dir=args.out,
                quant=args.quant,
                llama_cpp=args.llama_cpp,
                task=args.task,
                modality=args.modality,
                pooling=args.pooling,
                normalization=args.normalize,
            )
            print(result)
            return 0
        path = convert_gguf(
            args.source,
            args.out,
            quant=args.quant,
            llama_cpp=args.llama_cpp,
            task=args.task,
            modality=args.modality,
            pooling=args.pooling,
            normalization=args.normalize,
        )
        print(path)
        return 0
    if args.command == "verify":
        from fastretrieval.convert.verify import verify_converted

        report = verify_converted(args.converted, args.source, atol=args.atol)
        for key in ("max_abs_diff", "mean_abs_diff", "cosine"):
            print(f"{key}: {report[key]:.6g}")
        return 0 if report["passed"] else 1
    if args.command == "card":
        from pathlib import Path

        from fastretrieval.convert.card import render_card
        from fastretrieval.convert.verify import verify_manifest

        directory = Path(args.directory)
        contract = verify_manifest(directory, expected_source=args.source)
        sizes = {
            name: (directory / relative).stat().st_size / (1024**2)
            for name, relative in (
                ("int8", "onnx/model_quantized.onnx"),
                ("q4f16", "onnx/model_q4f16.onnx"),
            )
            if (directory / relative).exists()
        }
        kind = (
            "reranker"
            if contract.task in {"cross_encoder", "generative_reranker"}
            else "embedding"
        )
        destination = directory / "README.md"
        destination.write_text(
            render_card(args.source, sizes, kind=kind, contract=contract), encoding="utf-8"
        )
        print(destination)
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")
