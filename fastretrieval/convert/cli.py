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
    gguf.add_argument("--quant", default="Q4_K_M", help="llama-quantize type (default: Q4_K_M)")
    gguf.add_argument(
        "--llama-cpp",
        default=None,
        help="path to a llama.cpp checkout (default: $LLAMA_CPP_HOME)",
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
        from fastretrieval.convert.onnx import convert_onnx

        yes_no = (args.yes_token, args.no_token) if args.yes_no_head else None
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

        path = convert_gguf(
            args.source,
            args.out,
            quant=args.quant,
            llama_cpp=args.llama_cpp,
        )
        print(path)
        return 0
    raise AssertionError(f"unhandled command {args.command!r}")
