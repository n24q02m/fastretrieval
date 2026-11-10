"""Tests for fastretrieval.server.__main__."""

from __future__ import annotations

import os
import runpy
import sys
from unittest.mock import patch

import pytest

pytest.importorskip("uvicorn", reason="uvicorn belongs to the optional [server] extra")

from fastretrieval.server.__main__ import main  # noqa: E402

MODULE = "fastretrieval.server.api:app"


def test_main_uvicorn_import_error() -> None:
    """main() returns 1 with the [server] install hint when uvicorn is not installed."""
    with (
        patch.dict(sys.modules, {"uvicorn": None}),
        patch("sys.stderr") as mock_stderr,
        patch("sys.argv", ["fastretrieval-server"]),
    ):
        result = main()
        assert result == 1
        called_text = "".join(call.args[0] for call in mock_stderr.write.call_args_list)
        assert "uvicorn" in called_text
        assert 'pip install "fastretrieval[server]"' in called_text


def test_main_success() -> None:
    """main() returns 0 and calls uvicorn.run with correct arguments."""
    with (
        patch("uvicorn.run") as mock_run,
        patch(
            "sys.argv",
            [
                "fastretrieval-server",
                "--host",
                "0.0.0.0",
                "--port",
                "8888",
                "--log-level",
                "debug",
            ],
        ),
    ):
        result = main()
        assert result == 0
        mock_run.assert_called_once_with(MODULE, host="0.0.0.0", port=8888, log_level="debug")


def test_main_default_args() -> None:
    """main() uses default arguments when none are provided."""
    with patch("uvicorn.run") as mock_run, patch("sys.argv", ["fastretrieval-server"]):
        result = main()
        assert result == 0
        mock_run.assert_called_once_with(MODULE, host="127.0.0.1", port=9800, log_level="info")


def test_main_cache_dir_and_threads_flags_set_env() -> None:
    """--cache-dir/--threads flow into the env knobs the registry reads."""
    with (
        patch("uvicorn.run"),
        patch("sys.argv", ["fastretrieval-server", "--cache-dir", "/tmp/m", "--threads", "3"]),
    ):
        os.environ.pop("FASTRETRIEVAL_CACHE_PATH", None)
        os.environ.pop("FASTRETRIEVAL_THREADS", None)
        try:
            result = main()
            assert result == 0
            assert os.environ["FASTRETRIEVAL_CACHE_PATH"] == "/tmp/m"
            assert os.environ["FASTRETRIEVAL_THREADS"] == "3"
        finally:
            os.environ.pop("FASTRETRIEVAL_CACHE_PATH", None)
            os.environ.pop("FASTRETRIEVAL_THREADS", None)


def test_main_module_execution() -> None:
    """Test the if __name__ == "__main__": block."""
    with (
        patch("uvicorn.run"),
        patch("sys.exit") as mock_exit,
        patch("sys.argv", ["fastretrieval-server"]),
    ):
        runpy.run_module("fastretrieval.server.__main__", run_name="__main__")
        mock_exit.assert_called_once_with(0)
