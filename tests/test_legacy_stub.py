"""The final qwen3-embed release must stay a thin pointer at fastretrieval."""

import importlib.util
import sys
import tomllib
from pathlib import Path

import pytest

import fastretrieval

ROOT = Path(__file__).resolve().parent.parent
STUB_DIR = ROOT / "legacy" / "qwen3-embed"
STUB = tomllib.loads((STUB_DIR / "pyproject.toml").read_text(encoding="utf-8"))["project"]
MAIN = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

# The public names qwen3-embed 1.14.0 exported from `qwen3_embed`.
LEGACY_API = [
    "CustomModelSpec",
    "CustomRerankerSpec",
    "Device",
    "TextEmbedding",
    "TextCrossEncoder",
]


def test_stub_is_the_inactive_old_name_one_patch_above_its_last_release():
    assert STUB["name"] == "qwen3-embed"
    assert STUB["version"] == "1.14.1"
    assert "Development Status :: 7 - Inactive" in STUB["classifiers"]


def test_stub_depends_only_on_the_renamed_distribution():
    assert MAIN["name"] == "fastretrieval"
    assert [dep.split(">=")[0] for dep in STUB["dependencies"]] == [MAIN["name"]]
    assert STUB["optional-dependencies"] == {"gguf": [f"{MAIN['name']}[gguf]>=1.12.1"]}
    assert "gguf" in MAIN["optional-dependencies"]
    assert STUB["requires-python"] == MAIN["requires-python"]


def test_stub_metadata_names_no_personal_contact():
    assert all(set(author) == {"name"} for author in STUB["authors"])


def test_old_import_name_reexports_the_legacy_api(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "qwen3_embed", STUB_DIR / "qwen3_embed" / "__init__.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "qwen3_embed", module)

    with pytest.warns(DeprecationWarning, match="renamed to fastretrieval"):
        spec.loader.exec_module(module)

    assert module.__all__ == LEGACY_API
    for name in LEGACY_API:
        assert getattr(module, name) is getattr(fastretrieval, name)
