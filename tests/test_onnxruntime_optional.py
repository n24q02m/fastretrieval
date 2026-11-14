"""The package must import, and fail clearly at model load, without onnxruntime.

onnxruntime publishes no cp315 wheel, so pyproject.toml declares it only for
CPython < 3.15. On 3.15 every public facade still has to import, and building an
ONNX session has to say why it cannot, instead of an ``AttributeError`` on a
missing module. These tests reproduce that interpreter on every CI row by hiding
the module, so the branch is exercised even where onnxruntime is installed.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

import fastretrieval.common.onnx_model as onnx_model_module

ONNX_MODEL_PATH = Path(onnx_model_module.__file__)


def _load_without_onnxruntime(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Execute onnx_model.py as a fresh module while ``import onnxruntime`` fails."""
    # A ``None`` entry in sys.modules makes the import statement raise ImportError,
    # exactly as on an interpreter where the distribution is not installed.
    monkeypatch.setitem(sys.modules, "onnxruntime", None)
    spec = importlib.util.spec_from_file_location("_onnx_model_without_ort", ONNX_MODEL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # Registered like a real import: @dataclass resolves string annotations through
    # sys.modules[cls.__module__], which fails for an unregistered module before 3.15.
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def test_module_imports_without_onnxruntime(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_without_onnxruntime(monkeypatch)

    assert module.ort is None
    assert issubclass(module.OnnxModel, object)


def test_session_creation_names_the_missing_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load_without_onnxruntime(monkeypatch)

    with pytest.raises(ImportError, match=r"needs onnxruntime.*Python 3\.15"):
        module.OnnxModel()._instantiate_onnx_session(Path("model.onnx"))
