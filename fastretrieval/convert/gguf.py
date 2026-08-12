"""Convert a Hugging Face model to quantized GGUF through llama.cpp.

Both conversion steps run as llama.cpp subprocesses: one converts to F16 and
the other quantizes the result. Callers must provide a built llama.cpp
checkout through the argument or ``LLAMA_CPP_HOME``.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
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


def _promote_artifact_directory(staging_dir: Path, out_dir: Path) -> None:
    if out_dir.exists():
        if not out_dir.is_dir():
            raise FileExistsError(f"output path is not a directory: {out_dir}")
        if any(out_dir.iterdir()):
            raise FileExistsError(
                f"output directory is not empty: {out_dir}; choose a new output path"
            )
        out_dir.rmdir()
    os.replace(staging_dir, out_dir)


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
    task: str = "feature-extraction",
    modality: str = "text",
    pooling: str | None = None,
    normalization: bool | None = None,
) -> Path:
    """Download ``source``, convert it to F16 GGUF, then quantize it.

    Pooling and normalization are explicit contract metadata even though GGUF
    itself does not encode the post-processing policy.
    """

    if not _SAFE_NAME.match(quant):
        raise ValueError(
            f"{quant!r} is not a valid quantization type; expected something like Q4_K_M"
        )
    root = find_llama_cpp(llama_cpp)
    require_convert_deps("huggingface_hub", "gguf")
    from huggingface_hub import snapshot_download

    from fastretrieval.convert.manifest import write_manifest
    from fastretrieval.convert.profiles import resolve_profile

    profile = resolve_profile(source, task=task, modality=modality)
    contract = profile.build_contract(
        pooling=pooling,
        normalization=normalization,
        artifact_formats=("gguf",),
        quantization=quant,
        exporter_version="llama.cpp",
    )

    output_dir = Path(out_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stem = source.rstrip("/").split("/")[-1].lower()

    with tempfile.TemporaryDirectory(
        dir=output_dir.parent, prefix=f".{output_dir.name}.staging-"
    ) as tmp:
        temporary = Path(tmp)
        model_dir = temporary / "model"
        artifact_dir = temporary / "artifact"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        f16 = temporary / f"{stem}-f16.gguf"

        logger.info("downloading {}", source)
        source_path = Path(source)
        if source_path.is_dir():
            shutil.copytree(source_path, model_dir)
        else:
            snapshot_download(repo_id=source, local_dir=str(model_dir))

        logger.info("converting to GGUF F16")
        final = artifact_dir / f"{stem}-{quant.lower().replace('_', '-')}.gguf"
        _run(
            [
                sys.executable,
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

        for relative in contract.tokenizer_files:
            source_file = model_dir / relative
            if not source_file.is_file():
                raise FileNotFoundError(
                    f"{contract.model_id}: tokenizer file required by manifest is missing: "
                    f"{relative}"
                )
            destination = artifact_dir / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_file, destination)

        write_manifest(artifact_dir, contract, metadata={"quantization": quant})
        from fastretrieval.convert.verify import verify_manifest

        verify_manifest(artifact_dir, expected_source=contract.source)
        _promote_artifact_directory(artifact_dir, output_dir)

    result = output_dir / final.name
    logger.info("wrote {} ({:.1f} MB)", result, result.stat().st_size / (1024**2))
    return result
