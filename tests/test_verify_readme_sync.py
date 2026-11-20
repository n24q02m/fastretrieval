"""The README registry check must still run when a standard stream cannot be reconfigured.

CI, pre-commit and CD all run ``scripts/repo-bootstrap/verify_readme_sync.py``. A process
started with file descriptor 2 closed gets ``sys.stderr = None``, and a harness that swaps
stderr for an in-memory buffer leaves a stream without ``reconfigure``. Either way the
script must still print its report instead of dying with ``AttributeError`` before it
checks anything.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = REPO_ROOT / "scripts" / "repo-bootstrap"


def _load(name: str, monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / f"{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    # setitem removes the entry again on teardown, so the generic ``_lib`` name does not
    # leak into the rest of the session.
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "stderr",
    [None, io.StringIO()],
    ids=["stderr-closed", "stderr-without-reconfigure"],
)
def test_main_reports_when_stderr_cannot_be_reconfigured(
    monkeypatch: pytest.MonkeyPatch, stderr: io.StringIO | None
) -> None:
    # The script imports its sibling helper module as a top-level ``_lib``.
    _load("_lib", monkeypatch)
    verify_readme_sync = _load("verify_readme_sync", monkeypatch)
    stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", stderr)
    monkeypatch.setattr(sys, "argv", ["verify_readme_sync.py", f"--repo-root={REPO_ROOT}"])

    exit_code = verify_readme_sync.main()

    stdout.flush()
    report = stdout.buffer.getvalue().decode("utf-8")
    assert exit_code == 0, report
    assert "Summary:" in report
