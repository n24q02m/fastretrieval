import asyncio
import enum
import importlib.machinery
import importlib.util
import io
import sys
import types
from pathlib import Path

import pytest

from fastretrieval.convert.modal_backend import (
    BACKENDS,
    _download_artifact,
    _is_file_entry,
    _require_output_directory,
    _rewrite_result,
    _write_volume_file,
    resolve_backend,
    run_remote,
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


def test_download_artifact_rejects_absolute_path_traversal(tmp_path: Path):
    volume = _AsyncVolume({"/mnt/output/artifact//etc/passwd": b"bad"})

    with pytest.raises(ValueError, match="unsafe artifact path"):
        asyncio.run(_download_artifact(volume, "/mnt/output/artifact", tmp_path / "out"))


def test_rewrite_result_rejects_absolute_path_traversal():
    from fastretrieval.convert.modal_backend import _rewrite_result

    remote_result = {"path": "/mnt/output/artifact//etc/shadow"}
    rewritten = _rewrite_result(remote_result, "/mnt/output/artifact", Path("/safe/output"))
    # If the relative part starts with "/", _rewrite_result should ignore it and return the original dictionary unchanged.
    assert rewritten == remote_result
    assert rewritten["path"] == "/mnt/output/artifact//etc/shadow"


# ---------------------------------------------------------------------------
# Output directory guard
# ---------------------------------------------------------------------------
def test_output_directory_guard_creates_the_parent_of_a_new_output(tmp_path: Path):
    output = tmp_path / "nested" / "out"

    assert _require_output_directory(output) == output
    assert output.parent.is_dir()
    assert not output.exists()


def test_output_directory_guard_accepts_an_existing_empty_directory(tmp_path: Path):
    output = tmp_path / "out"
    output.mkdir()

    assert _require_output_directory(str(output)) == output


def test_output_directory_guard_rejects_a_file(tmp_path: Path):
    output = tmp_path / "out"
    output.write_text("x", encoding="utf-8")

    with pytest.raises(FileExistsError, match="output path is not a directory"):
        _require_output_directory(output)


def test_output_directory_guard_rejects_a_non_empty_directory(tmp_path: Path):
    output = tmp_path / "out"
    output.mkdir()
    (output / "old.bin").write_bytes(b"old")

    with pytest.raises(FileExistsError, match="output directory is not empty"):
        _require_output_directory(output)


# ---------------------------------------------------------------------------
# Volume entry and file transfer helpers
# ---------------------------------------------------------------------------
# Boundary (b): stands in for Modal's FileEntryType; the modal SDK is not a test dependency.
class _EntryType(enum.Enum):
    FILE = 1
    DIRECTORY = 2


# Boundary (b): a Modal Volume listing entry; the modal SDK is not a test dependency.
class _TypedEntry:
    def __init__(self, path: str, entry_type: object):
        self.path = path
        self.type = entry_type


@pytest.mark.parametrize(
    ("entry", "is_file"),
    [
        (types.SimpleNamespace(path="a"), True),
        (_TypedEntry("a", _EntryType.FILE), True),
        (_TypedEntry("a", _EntryType.DIRECTORY), False),
        (_TypedEntry("a", "file"), True),
        (_TypedEntry("a", "directory"), False),
    ],
    ids=["untyped", "enum-file", "enum-directory", "str-file", "str-directory"],
)
def test_file_entry_detection_accepts_enum_and_string_types(entry, is_file):
    assert _is_file_entry(entry) is is_file


# Boundary (b): a Modal Volume whose read_file returns one payload shape; no modal SDK in tests.
class _ReadVolume:
    def __init__(self, data: object):
        self.data = data
        self.paths: list[str] = []

    def read_file(self, path: str):
        self.paths.append(path)
        return self.data


async def _async_chunks(*chunks: bytes):
    for chunk in chunks:
        yield chunk


@pytest.mark.parametrize(
    "make_data",
    [
        lambda: b"payload",
        lambda: bytearray(b"payload"),
        lambda: memoryview(b"payload"),
        lambda: io.BytesIO(b"payload"),
        lambda: _async_chunks(b"pay", b"load"),
        lambda: [b"pay", b"load"],
    ],
    ids=["bytes", "bytearray", "memoryview", "readable", "async-chunks", "sync-chunks"],
)
def test_volume_file_is_written_from_every_supported_read_shape(tmp_path: Path, make_data):
    volume = _ReadVolume(make_data())
    destination = tmp_path / "nested" / "model.onnx"

    asyncio.run(_write_volume_file(volume, "/mnt/output/artifact/model.onnx", destination))

    assert volume.paths == ["/mnt/output/artifact/model.onnx"]
    assert destination.read_bytes() == b"payload"


# Boundary (b): a Modal Volume with synchronous calls; the modal SDK is not a test dependency.
class _SyncVolume:
    """A volume whose reload/iterdir are plain synchronous calls."""

    def __init__(self, entries: list, files: dict[str, bytes]):
        self.entries = entries
        self.files = files
        self.reloads = 0
        self.iterdir_calls: list[tuple[str, bool]] = []

    def reload(self):
        self.reloads += 1

    def iterdir(self, root: str, *, recursive: bool):
        self.iterdir_calls.append((root, recursive))
        return list(self.entries)

    def read_file(self, path: str):
        return self.files[path]


def test_download_artifact_supports_a_synchronous_volume_and_skips_directories(tmp_path: Path):
    root = "/mnt/output/artifact"
    volume = _SyncVolume(
        [
            _TypedEntry(f"{root}/onnx", _EntryType.DIRECTORY),
            _TypedEntry(f"{root}/onnx/model_quantized.onnx", _EntryType.FILE),
            _TypedEntry(f"{root}/config.json", _EntryType.FILE),
        ],
        {
            f"{root}/onnx/model_quantized.onnx": b"int8",
            f"{root}/config.json": b"{}",
        },
    )

    files = asyncio.run(_download_artifact(volume, f"{root}/", tmp_path / "out"))

    assert files == ["config.json", "onnx/model_quantized.onnx"]
    assert volume.reloads == 1
    assert volume.iterdir_calls == [(root, True)]
    assert (tmp_path / "out" / "onnx" / "model_quantized.onnx").read_bytes() == b"int8"
    assert (tmp_path / "out" / "config.json").read_bytes() == b"{}"


@pytest.mark.parametrize(
    ("path", "message"),
    [
        ("/mnt/output/other/model.onnx", "outside the artifact root"),
        ("/mnt/output/artifact-sibling/model.onnx", "outside the artifact root"),
        ("/mnt/output/artifact/", "unsafe artifact path"),
    ],
)
def test_download_artifact_rejects_paths_outside_or_at_the_root(tmp_path: Path, path, message):
    volume = _SyncVolume([_TypedEntry(path, _EntryType.FILE)], {path: b"x"})

    with pytest.raises(ValueError, match=message):
        asyncio.run(_download_artifact(volume, "/mnt/output/artifact", tmp_path / "out"))


def test_download_artifact_fails_when_the_volume_holds_no_files(tmp_path: Path):
    volume = _SyncVolume([_TypedEntry("/mnt/output/artifact/onnx", _EntryType.DIRECTORY)], {})

    with pytest.raises(RuntimeError, match="returned no files under /mnt/output/artifact"):
        asyncio.run(_download_artifact(volume, "/mnt/output/artifact", tmp_path / "out"))


@pytest.mark.parametrize(
    "result",
    [
        ["not", "a", "dict"],
        {"path": 7},
        {"files": 3},
        {"path": "/elsewhere/model.gguf"},
        {"path": "/mnt/output/artifact/../escape.gguf"},
        {"path": "/mnt/output/artifact/"},
    ],
    ids=["list", "non-str-path", "no-path", "outside-root", "traversal", "root-only"],
)
def test_rewrite_result_leaves_unmappable_results_untouched(result):
    assert _rewrite_result(result, "/mnt/output/artifact", Path("out")) is result


def test_rewrite_result_maps_the_remote_path_into_the_local_output(tmp_path: Path):
    remote = {"path": "/mnt/output/artifact/sub/model-q4-k-m.gguf", "size": 1}

    rewritten = _rewrite_result(remote, "/mnt/output/artifact/", tmp_path)

    assert rewritten == {"path": str(tmp_path / "sub" / "model-q4-k-m.gguf"), "size": 1}
    assert remote["path"] == "/mnt/output/artifact/sub/model-q4-k-m.gguf"


# ---------------------------------------------------------------------------
# run_remote argument validation (fails before any Modal API call)
# ---------------------------------------------------------------------------
@pytest.fixture
def modal_installed(monkeypatch):
    # Boundary (b): the modal SDK is not a test dependency; validation only needs
    # find_spec("modal") to succeed, and every case below fails before ``import modal``.
    module = types.ModuleType("modal")
    module.__spec__ = importlib.machinery.ModuleSpec("modal", None)
    monkeypatch.setitem(sys.modules, "modal", module)
    return module


def test_run_remote_rejects_an_unsupported_command():
    with pytest.raises(ValueError, match="unsupported remote command 'verify'"):
        run_remote("verify", source="acme/tiny", out_dir="out")


@pytest.mark.usefixtures("modal_installed")
@pytest.mark.parametrize("payload", [{}, {"out_dir": ""}, {"out": None}])
def test_run_remote_requires_an_output_directory(payload):
    with pytest.raises(TypeError, match="remote conversion requires out_dir"):
        run_remote("onnx", source="acme/tiny", **payload)


@pytest.mark.usefixtures("modal_installed")
@pytest.mark.parametrize("source", [None, "", 42])
def test_run_remote_requires_a_source_string(tmp_path: Path, source):
    with pytest.raises(TypeError, match="remote conversion requires a source string"):
        run_remote("onnx", source=source, out=tmp_path / "out")


@pytest.mark.usefixtures("modal_installed")
def test_run_remote_refuses_a_non_empty_output_directory(tmp_path: Path):
    output = tmp_path / "out"
    output.mkdir()
    (output / "old.bin").write_bytes(b"old")

    with pytest.raises(FileExistsError, match="output directory is not empty"):
        run_remote("onnx", source="acme/tiny", out_dir=output)


@pytest.mark.usefixtures("modal_installed")
def test_remote_gguf_requires_a_llama_cpp_checkout_argument(tmp_path: Path):
    with pytest.raises(ValueError, match="requires --llama-cpp"):
        run_remote("gguf", source="acme/tiny", out_dir=tmp_path / "out")


@pytest.mark.usefixtures("modal_installed")
def test_remote_gguf_requires_the_llama_cpp_checkout_to_exist(tmp_path: Path):
    with pytest.raises(FileNotFoundError, match="llama.cpp checkout not found at"):
        run_remote(
            "gguf", source="acme/tiny", out_dir=tmp_path / "out", llama_cpp=tmp_path / "absent"
        )
