"""Review Focus 4 boundary: the model runtime and the HTTP surface have one owner.

The runtime lives here, so this repo must not import it back from a consumer
(the ``n24q02m-hull`` dist that used to ship this server, its legacy
``qwen3-embed`` dist name, or any of hull's import packages). The reverse
direction is asserted in hull's own ``test_embedding_boundary.py``.
"""

from __future__ import annotations

import ast
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("fastapi", reason="fastapi belongs to the optional [server] extra")

ROOT = Path(__file__).resolve().parents[1]

# Import names owned by the hull dist (the server used to live there) and the
# legacy dist name this fork was published under. Importing any of them from
# here would mean the runtime still depends on a consumer.
FORBIDDEN = ("hull_core", "hull_web", "hull_embedding_daemon", "qwen3_embed")


def _imported_roots(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def test_package_never_imports_a_consumer() -> None:
    offenders = []
    files = sorted((ROOT / "fastretrieval").rglob("*.py"))
    assert files, "no sources found under fastretrieval/"
    for path in files:
        for root in sorted(_imported_roots(path) & set(FORBIDDEN)):
            offenders.append(f"{path.relative_to(ROOT)}: {root}")
    assert not offenders, "the runtime must not import its consumers: " + ", ".join(offenders)


def test_library_import_does_not_pull_the_server_extra() -> None:
    """``import fastretrieval`` must work without ``[server]`` installed."""
    code = "import sys; import fastretrieval; print('fastapi' in sys.modules, 'uvicorn' in sys.modules)"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "False False", proc.stdout


def test_server_import_does_not_pull_hull_modules() -> None:
    code = (
        "import sys; import fastretrieval.server.api; "
        "print([m for m in sys.modules if m.split('.')[0] in "
        f"{FORBIDDEN!r}])"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip() == "[]", proc.stdout
