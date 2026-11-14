#!/usr/bin/env python3
"""Derive the CI Python matrix from the declared ``requires-python`` floor.

A hand-written matrix is checked against nothing, so it rots: the floor moves in
``pyproject.toml`` and the workflow keeps testing a version the package no longer
claims, or stops testing one it just started claiming. The list therefore lives
here as data derived from ``pyproject.toml``, and ``tests/test_python_support.py``
asserts that ``.github/workflows/ci.yml`` still agrees with it.

Usage::

    python scripts/ci_versions.py
    ["3.11", "3.12", "3.13", "3.14", "3.15"]

The output is a JSON array, ready for ``strategy.matrix.python`` through
``fromJSON``.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"

# ---------------------------------------------------------------------------
# Ceiling
# ---------------------------------------------------------------------------
# The top of the window is the newest CPython minor with a final release that this
# package can install. A row that cannot resolve is worse than an absent row: it
# turns every PR red for a reason no PR caused.
#
# 3.15 (3.15.0 final, October 2026) is that minor. It installs although
# onnxruntime, a hard dependency everywhere else, publishes no cp315 wheel and no
# sdist (through 1.32.0, nightlies included): pyproject.toml declares onnxruntime
# with `python_version < '3.15'`, and numpy with a cp315-capable floor there, so
# `uv sync` resolves on 3.15 and the ONNX backend reports the missing runtime at
# model load instead of the install failing. The 3.15 row therefore runs the full
# unit suite, and the import-without-runtime path is asserted on its own.
#
# Raise NEWEST_SUPPORTED_MINOR when CPython 3.16 has a final release and its
# wheels resolve. The reason is exported (and asserted) so the ceiling is never
# just a silent absence.
NEWEST_SUPPORTED_MINOR = (3, 15)

CEILING_REASON = (
    "CPython 3.16 has no final release yet (3.15.0 shipped in October 2026), so "
    "3.15 is the newest minor a published package can claim; on 3.15 onnxruntime "
    "1.32.0 still has no cp315 wheel, which pyproject.toml handles with a marker."
)

# The Q34 contract caps the matrix at five rows. Fail loudly instead of silently
# truncating when requires-python drops far enough that the window would exceed
# the cap.
MAX_MATRIX_ROWS = 5

_CLAUSE_RE = re.compile(r"^(>=|>|<=|<|==)\s*3\.(\d{1,2})(?:\.\d+)?$")


def read_requires_python(pyproject_path: Path = PYPROJECT_PATH) -> str:
    """Return ``project.requires-python`` from a pyproject.toml."""
    with pyproject_path.open("rb") as handle:
        document = tomllib.load(handle)
    try:
        declared = document["project"]["requires-python"]
    except KeyError as exc:
        raise ValueError(f"{pyproject_path} declares no project.requires-python") from exc
    if not isinstance(declared, str):
        raise ValueError(f"{pyproject_path}: requires-python is not a string")
    return declared


def parse_window(requires_python: str) -> tuple[int, int | None]:
    """Split a requires-python spec into ``(floor, exclusive_ceiling)`` minors.

    ``exclusive_ceiling`` is ``None`` when the spec declares no upper bound. Only
    CPython 3.x clauses are understood; anything else raises rather than guessing.
    """
    floor: int | None = None
    exclusive_ceiling: int | None = None
    for clause in requires_python.split(","):
        clause = clause.strip()
        if not clause:
            continue
        match = _CLAUSE_RE.match(clause)
        if match is None:
            raise ValueError(f"unsupported requires-python clause: {clause!r}")
        operator, minor = match.group(1), int(match.group(2))
        if operator in (">=", ">"):
            candidate = minor if operator == ">=" else minor + 1
            floor = candidate if floor is None else max(floor, candidate)
        else:
            candidate = minor if operator == "<" else minor + 1
            exclusive_ceiling = (
                candidate if exclusive_ceiling is None else min(exclusive_ceiling, candidate)
            )
        if operator == "==":
            floor = minor if floor is None else max(floor, minor)
    if floor is None:
        raise ValueError(f"requires-python {requires_python!r} declares no lower bound")
    return floor, exclusive_ceiling


def supported_minors(
    *,
    pyproject_path: Path = PYPROJECT_PATH,
    newest_minor: int | None = None,
) -> list[str]:
    """Return the ``X.Y`` minors the CI matrix must cover, in ascending order."""
    floor, exclusive_ceiling = parse_window(read_requires_python(pyproject_path))
    ceiling = NEWEST_SUPPORTED_MINOR[1] if newest_minor is None else newest_minor
    if exclusive_ceiling is not None:
        ceiling = min(ceiling, exclusive_ceiling - 1)
    if ceiling < floor:
        raise ValueError(f"empty support window: floor 3.{floor} exceeds ceiling 3.{ceiling}")
    rows = ceiling - floor + 1
    if rows > MAX_MATRIX_ROWS:
        raise ValueError(
            f"support window 3.{floor}-3.{ceiling} needs {rows} rows, "
            f"over the {MAX_MATRIX_ROWS}-row cap; raise the floor in pyproject.toml"
        )
    return [f"3.{minor}" for minor in range(floor, ceiling + 1)]


def main() -> int:
    print(json.dumps(supported_minors()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
