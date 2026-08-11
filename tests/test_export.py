"""Tests for the optional HF-id to ONNX export helper."""

import importlib.util
from unittest.mock import MagicMock, patch

import pytest

from qwen3_embed.export import export_to_onnx

_HAS_OPTIMUM = importlib.util.find_spec("optimum") is not None


def test_export_without_extra_raises_helpful_error(tmp_path):
    with (
        patch.dict("sys.modules", {"optimum": None}),
        pytest.raises(ImportError, match=r"optimum\[exporters\]"),
    ):
        export_to_onnx("sentence-transformers/all-MiniLM-L6-v2", str(tmp_path))


def test_export_mocked(tmp_path):
    """Test export_to_onnx by mocking optimum to ensure coverage without heavy deps."""
    mock_main_export = MagicMock()
    # We mock the entire path to avoid ImportError when optimum is not installed
    with patch.dict(
        "sys.modules",
        {
            "optimum": MagicMock(),
            "optimum.exporters": MagicMock(),
            "optimum.exporters.onnx": MagicMock(main_export=mock_main_export),
        },
    ):
        # We also need to patch where it's imported in the module to avoid the try/except block failing
        # but wait, the try/except block DOES 'from optimum.exporters.onnx import main_export'.
        # If we patched sys.modules, that import should succeed and return our mock.

        model_id = "mock/model"
        output_dir = str(tmp_path / "exported")

        result = export_to_onnx(model_id, output_dir, task="test-task")

        assert result == output_dir
        assert (tmp_path / "exported").exists()
        mock_main_export.assert_called_once_with(model_id, output=output_dir, task="test-task")


@pytest.mark.integration
@pytest.mark.skipif(not _HAS_OPTIMUM, reason="requires the [export] extra")
def test_export_minilm_roundtrip(tmp_path):
    import onnxruntime as ort  # noqa: F401

    out = export_to_onnx("sentence-transformers/all-MiniLM-L6-v2", str(tmp_path))
    assert (tmp_path / "model.onnx").exists() or (tmp_path / "onnx" / "model.onnx").exists()
    assert out == str(tmp_path)
