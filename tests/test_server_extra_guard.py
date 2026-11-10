"""The ``[server]`` extra guard: clear ImportError / exit 1 with an install hint."""

from __future__ import annotations

import importlib
import subprocess
import sys
from unittest.mock import patch

import pytest

from fastretrieval.server import _extra
from fastretrieval.server.__main__ import main

INSTALL_HINT = 'pip install "fastretrieval[server]"'


def test_missing_server_modules_empty_when_all_present() -> None:
    assert _extra.missing_server_modules(find_spec=lambda name: object()) == []


def test_missing_server_modules_lists_absent_modules_in_order() -> None:
    absent = {"uvicorn", "fastapi"}
    found = _extra.missing_server_modules(
        find_spec=lambda name: None if name in absent else object()
    )
    assert found == ["fastapi", "uvicorn"]


def test_require_server_extra_raises_with_install_hint() -> None:
    with pytest.raises(ImportError) as exc_info:
        _extra.require_server_extra(
            find_spec=lambda name: None if name in {"fastapi", "pydantic"} else object()
        )
    message = str(exc_info.value)
    assert INSTALL_HINT in message
    assert "fastapi, pydantic" in message
    assert exc_info.value.name == "fastapi"


def test_require_server_extra_passes_when_all_present() -> None:
    _extra.require_server_extra(find_spec=lambda name: object())


@pytest.mark.parametrize("module", ["fastapi", "pydantic"])
def test_import_api_without_extra_raises_install_hint(
    monkeypatch: pytest.MonkeyPatch, module: str
) -> None:
    # A None entry in sys.modules makes find_spec() report the module as absent.
    monkeypatch.setitem(sys.modules, module, None)
    monkeypatch.delitem(sys.modules, "fastretrieval.server.api", raising=False)

    with pytest.raises(ImportError) as exc_info:
        importlib.import_module("fastretrieval.server.api")

    assert INSTALL_HINT in str(exc_info.value)
    assert module in str(exc_info.value)


def test_package_import_stays_light_without_extra() -> None:
    # __version__ must stay readable from a base install (no [server] extra).
    # A None entry in sys.modules makes both ``import`` and ``find_spec`` treat
    # the module as absent; a subprocess keeps that state out of this session.
    code = (
        "import sys\n"
        "sys.modules['fastapi'] = None\n"
        "sys.modules['uvicorn'] = None\n"
        "sys.modules['pydantic'] = None\n"
        "import fastretrieval.server as s\n"
        "print(s.__version__)\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip()


@pytest.mark.parametrize("module", ["fastapi", "uvicorn", "pydantic"])
def test_main_returns_1_with_hint_when_extra_missing(
    monkeypatch: pytest.MonkeyPatch, module: str
) -> None:
    import uvicorn

    # Patch the real module object first; sys.modules["uvicorn"] may be nulled below.
    with (
        patch.object(uvicorn, "run") as mock_run,
        patch("sys.stderr") as mock_stderr,
        patch("sys.argv", ["fastretrieval-server"]),
    ):
        monkeypatch.setitem(sys.modules, module, None)
        result = main()
    assert result == 1
    mock_run.assert_not_called()
    written = "".join(call.args[0] for call in mock_stderr.write.call_args_list)
    assert INSTALL_HINT in written
    assert module in written
