"""Convert a Hugging Face model to quantized GGUF through llama.cpp.

Both conversion steps run as llama.cpp subprocesses: one converts to F16 and
the other quantizes the result. Callers must provide a built llama.cpp
checkout through the argument or ``LLAMA_CPP_HOME``.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

from loguru import logger

from fastretrieval.convert import require_convert_deps

_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")

_BUILD_HINT = (
    "Set LLAMA_CPP_HOME to a built llama.cpp checkout, or pass --llama-cpp:\n"
    "  git clone --depth 1 https://github.com/ggml-org/llama.cpp\n"
    "  cd llama.cpp && cmake -B build -DGGML_NATIVE=OFF \\\n"
    "    && cmake --build build --target llama-quantize -j"
)


def find_llama_cpp(explicit: str | None = None) -> Path:
    """Locate llama.cpp and verify the converter and quantizer are present."""

    raw = explicit or os.environ.get("LLAMA_CPP_HOME")
    if not raw:
        raise FileNotFoundError(f"llama.cpp checkout not found. {_BUILD_HINT}")
    root = Path(raw)
    if not root.is_dir():
        raise FileNotFoundError(f"llama.cpp checkout not found at {root}. {_BUILD_HINT}")

    converter = root / "convert_hf_to_gguf.py"
    quantizer = root / "build" / "bin" / "llama-quantize"
    for path, what in (
        (converter, "convert_hf_to_gguf.py"),
        (quantizer, "llama-quantize"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{what} is missing under {root}. {_BUILD_HINT}")
    return root


def _run(cmd: list[str], what: str) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        logger.error("{} failed:\n{}", what, result.stderr)
        raise RuntimeError(f"{what} exited with code {result.returncode}")


def convert_gguf(
    source: str,
    out_dir: str | Path,
    *,
    quant: str = "Q4_K_M",
    llama_cpp: str | None = None,
) -> Path:
    """Download ``source``, convert it to F16 GGUF, then quantize it."""

    if not _SAFE_NAME.match(quant):
        raise ValueError(
            f"{quant!r} is not a valid quantization type; expected something like Q4_K_M"
        )
    root = find_llama_cpp(llama_cpp)
    require_convert_deps("huggingface_hub", "gguf")
    from huggingface_hub import snapshot_download

    output_dir = Path(out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = source.rstrip("/").split("/")[-1].lower()
    final = output_dir / f"{stem}-{quant.lower().replace('_', '-')}.gguf"

    with tempfile.TemporaryDirectory() as tmp:
        model_dir = Path(tmp) / "model"
        f16 = Path(tmp) / f"{stem}-f16.gguf"

        logger.info("downloading {}", source)
        snapshot_download(repo_id=source, local_dir=str(model_dir))

        logger.info("converting to GGUF F16")
        _run(
            [
                "python",
                str(root / "convert_hf_to_gguf.py"),
                str(model_dir.resolve()),
                "--outfile",
                str(f16.resolve()),
                "--outtype",
                "f16",
            ],
            "convert_hf_to_gguf.py",
        )

        logger.info("quantizing to {}", quant)
        _run(
            [
                str(root / "build" / "bin" / "llama-quantize"),
                str(f16.resolve()),
                str(final.resolve()),
                quant,
            ],
            "llama-quantize",
        )

    logger.info("wrote {} ({:.1f} MB)", final, final.stat().st_size / (1024**2))
    return final
