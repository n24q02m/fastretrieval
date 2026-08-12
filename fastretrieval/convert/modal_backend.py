"""Run one conversion in an optional Modal CPU container.

The local backend remains the default. The Modal path is deliberately explicit:
the source package and any local model/llama.cpp checkout are mounted into the
container, while the finished artifact is transferred through an ephemeral
Volume instead of being serialized through the function return value.
"""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
from pathlib import Path, PurePosixPath
from typing import Any

BACKENDS = ("local", "modal")

_IMAGE_PACKAGES = (
    "torch>=2.4",
    "transformers>=4.47",
    "optimum-onnx[onnxruntime]>=0.1.0",
    "onnx>=1.17",
    "onnxruntime[quantization]>=1.21",
    "onnx-ir>=0.2.0",
    "onnxconverter-common>=1.14",
    "huggingface-hub>=0.30",
    "gguf>=0.6",
    "loguru>=0.7",
)
_REMOTE_OUTPUT = "/mnt/output/artifact"


def resolve_backend(name: str | None) -> str:
    """Return a supported backend name; ``None`` means local."""
    if name is None:
        return "local"
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; pick from {', '.join(BACKENDS)}")
    return name


def _require_output_directory(path: str | Path) -> Path:
    output = Path(path)
    if output.exists():
        if not output.is_dir():
            raise FileExistsError(f"output path is not a directory: {output}")
        if any(output.iterdir()):
            raise FileExistsError(
                f"output directory is not empty: {output}; choose a new output path"
            )
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
    return output


def _is_file_entry(entry: Any) -> bool:
    entry_type = getattr(entry, "type", None)
    if entry_type is None:
        return True
    name = getattr(entry_type, "name", str(entry_type)).upper()
    return "FILE" in name


async def _write_volume_file(volume: Any, remote_path: str, destination: Path) -> None:
    data = volume.read_file(remote_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("wb") as handle:
        if isinstance(data, (bytes, bytearray, memoryview)):
            handle.write(data)
        elif hasattr(data, "read"):
            while chunk := data.read(1024 * 1024):
                handle.write(chunk)
        elif hasattr(data, "__aiter__"):
            async for chunk in data:
                handle.write(chunk)
        else:
            for chunk in data:
                handle.write(chunk)


async def _download_artifact(volume: Any, remote_root: str, output_dir: Path) -> list[str]:
    reload_result = volume.reload()
    if inspect.isawaitable(reload_result):
        await reload_result
    root = remote_root.rstrip("/")
    downloaded: list[str] = []
    entries = volume.iterdir(root, recursive=True)
    async for entry in _async_entries(entries):
        if not _is_file_entry(entry):
            continue
        remote_path = str(entry.path)
        prefix = f"{root}/"
        if not remote_path.startswith(prefix):
            raise ValueError(f"Modal returned a file outside the artifact root: {remote_path}")
        relative = PurePosixPath(remote_path[len(prefix) :])
        if not relative.parts or ".." in relative.parts:
            raise ValueError(f"Modal returned an unsafe artifact path: {remote_path}")
        destination = output_dir.joinpath(*relative.parts)
        await _write_volume_file(volume, remote_path, destination)
        downloaded.append(relative.as_posix())
    if not downloaded:
        raise RuntimeError(f"Modal conversion returned no files under {remote_root}")
    return sorted(downloaded)


async def _async_entries(entries: Any):
    if hasattr(entries, "__aiter__"):
        async for entry in entries:
            yield entry
    else:
        for entry in entries:
            yield entry


def _rewrite_result(result: Any, remote_root: str, output_dir: Path) -> Any:
    if not isinstance(result, dict) or not isinstance(result.get("path"), str):
        return result
    remote_path = result["path"]
    prefix = f"{remote_root.rstrip('/')}/"
    if remote_path.startswith(prefix):
        relative = PurePosixPath(remote_path[len(prefix) :])
        if relative.parts and ".." not in relative.parts:
            result = dict(result)
            result["path"] = str(output_dir.joinpath(*relative.parts))
    return result


def run_remote(command: str, **kwargs: Any) -> dict[str, Any]:
    """Run one ONNX or GGUF conversion remotely and download its artifact.

    ``source`` may be a HuggingFace id or a local model directory. Local GGUF
    conversion additionally requires a local, built llama.cpp checkout because
    this backend does not silently build native code in the container.
    """
    if command not in {"onnx", "gguf"}:
        raise ValueError(f"unsupported remote command {command!r}; pick from onnx, gguf")
    if importlib.util.find_spec("modal") is None:
        raise ImportError(
            "the modal backend needs the modal package: pip install modal, then `modal token new`"
        )

    payload = dict(kwargs)
    if "out_dir" not in payload and "out" in payload:
        payload["out_dir"] = payload.pop("out")
    if not payload.get("out_dir"):
        raise TypeError("remote conversion requires out_dir")
    output_dir = _require_output_directory(payload["out_dir"])

    source = payload.get("source")
    if not isinstance(source, str) or not source:
        raise TypeError("remote conversion requires a source string")
    source_path = Path(source)
    local_source = source_path.resolve() if source_path.is_dir() else None

    local_llama = None
    if command == "gguf":
        raw_llama = payload.get("llama_cpp")
        if not raw_llama:
            raise ValueError(
                "Modal GGUF conversion requires --llama-cpp with a built llama.cpp checkout"
            )
        llama_path = Path(raw_llama)
        if not llama_path.is_dir():
            raise FileNotFoundError(f"llama.cpp checkout not found at {llama_path}")
        local_llama = llama_path.resolve()

    import modal

    image = (
        modal.Image.debian_slim(python_version="3.12")
        .uv_pip_install(*_IMAGE_PACKAGES)
        .add_local_python_source("fastretrieval")
    )
    if local_source is not None:
        image = image.add_local_dir(str(local_source), "/mnt/input-model", copy=False)
        payload["source"] = "/mnt/input-model"
    if local_llama is not None:
        image = image.add_local_dir(str(local_llama), "/mnt/llama-cpp", copy=False)
        payload["llama_cpp"] = "/mnt/llama-cpp"
    payload["out_dir"] = _REMOTE_OUTPUT

    async def execute() -> dict[str, Any]:
        async with modal.Volume.ephemeral() as volume:
            app = modal.App("fastretrieval-convert")

            @app.function(
                image=image,
                memory=32768,
                cpu=4.0,
                timeout=3600,
                volumes={"/mnt/output": volume},
            )
            async def _convert(
                remote_command: str, remote_payload: dict[str, Any]
            ) -> dict[str, Any]:
                if remote_command == "onnx":
                    from fastretrieval.convert.onnx import convert_onnx

                    result = convert_onnx(**remote_payload)
                    await volume.commit()
                    return {"result": result}
                if remote_command == "gguf":
                    from fastretrieval.convert.gguf import convert_gguf

                    result = convert_gguf(**remote_payload)
                    await volume.commit()
                    return {"result": {"path": str(result)}}
                raise ValueError(f"unsupported remote command {remote_command!r}")

            async with app.run():
                remote_result = await _convert.remote.aio(command, payload)

            files = await _download_artifact(volume, _REMOTE_OUTPUT, output_dir)

        return {
            "backend": "modal",
            "command": command,
            "out_dir": str(output_dir),
            "files": files,
            "result": _rewrite_result(remote_result["result"], _REMOTE_OUTPUT, output_dir),
        }

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(execute())
    raise RuntimeError("run_remote is synchronous and cannot run inside an active event loop")


__all__ = ["BACKENDS", "resolve_backend", "run_remote"]
