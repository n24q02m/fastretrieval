import asyncio
import importlib.util
from pathlib import Path

import pytest

from fastretrieval.convert.modal_backend import (
    BACKENDS,
    _download_artifact,
    resolve_backend,
)


def test_local_is_the_default():
    assert resolve_backend(None) == "local"


def test_modal_backend_is_selectable():
    assert resolve_backend("modal") == "modal"


def test_unknown_backend_lists_the_valid_ones():
    with pytest.raises(ValueError, match="local"):
        resolve_backend("ec2")


def test_modal_backend_explains_itself_when_modal_is_absent(monkeypatch):
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name == "modal" else importlib.util.find_spec(name),
    )
    from fastretrieval.convert.modal_backend import run_remote

    with pytest.raises(ImportError, match="modal"):
        run_remote("onnx", source="acme/tiny", out="/tmp/out")


def test_backends_tuple_is_the_single_source_of_truth():
    assert BACKENDS == ("local", "modal")


class _Entry:
    type = "FILE"

    def __init__(self, path: str):
        self.path = path


class _AsyncVolume:
    def __init__(self, files: dict[str, bytes]):
        self.files = files
        self.reloaded = False

    async def reload(self):
        self.reloaded = True

    async def iterdir(self, _root: str, *, recursive: bool):
        for path in self.files:
            yield _Entry(path)

    async def read_file(self, path: str):
        yield self.files[path]


def test_download_artifact_awaits_current_modal_volume_api(tmp_path: Path):
    volume = _AsyncVolume({"/mnt/output/artifact/onnx/model.onnx": b"onnx"})

    files = asyncio.run(_download_artifact(volume, "/mnt/output/artifact", tmp_path / "out"))

    assert volume.reloaded is True
    assert files == ["onnx/model.onnx"]
    assert (tmp_path / "out" / "onnx" / "model.onnx").read_bytes() == b"onnx"


def test_download_artifact_rejects_path_traversal(tmp_path: Path):
    volume = _AsyncVolume({"/mnt/output/artifact/../escape": b"bad"})

    with pytest.raises(ValueError, match="unsafe artifact path"):
        asyncio.run(_download_artifact(volume, "/mnt/output/artifact", tmp_path / "out"))
