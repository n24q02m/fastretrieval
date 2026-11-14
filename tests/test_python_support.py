"""Contract checks tying the CI matrix to the declared ``requires-python`` floor.

Every hard-coded ``3.1x`` list drifts eventually: the floor moves in
``pyproject.toml`` and the workflow, the classifiers, the ruff target and the
runtime window are all left behind. The support window is therefore derived once
in ``scripts/ci_versions.py`` and asserted here against each place that has to
agree with it.
"""

from __future__ import annotations

import importlib.util
import re
import sys
import tomllib
from importlib.metadata import metadata
from pathlib import Path

import pytest
from packaging.requirements import Requirement

REPO_ROOT = Path(__file__).resolve().parents[1]
PYPROJECT_PATH = REPO_ROOT / "pyproject.toml"
CI_WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci.yml"
CI_VERSIONS_PATH = REPO_ROOT / "scripts" / "ci_versions.py"


def _load_ci_versions():
    spec = importlib.util.spec_from_file_location("ci_versions", CI_VERSIONS_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["ci_versions"] = module
    spec.loader.exec_module(module)
    return module


ci_versions = _load_ci_versions()
SUPPORTED = ci_versions.supported_minors()


def _pyproject() -> dict:
    with PYPROJECT_PATH.open("rb") as handle:
        return tomllib.load(handle)


def _ci_matrix_minors() -> list[str]:
    """Read the ``python: [...]`` matrix list out of the lint-and-test job."""
    workflow = CI_WORKFLOW_PATH.read_text(encoding="utf-8")
    matches = re.findall(r"^\s+python:\s*\[(.*?)\]\s*$", workflow, re.MULTILINE)
    assert len(matches) == 1, (
        f"expected exactly one `python: [...]` matrix in {CI_WORKFLOW_PATH.name}, "
        f"found {len(matches)}"
    )
    return [item.strip().strip("\"'") for item in matches[0].split(",") if item.strip()]


def test_support_window_is_derived_from_the_declared_floor():
    floor, _ = ci_versions.parse_window(ci_versions.read_requires_python())
    assert SUPPORTED[0] == f"3.{floor}"
    assert [f"3.{minor}" for minor in range(floor, floor + len(SUPPORTED))] == SUPPORTED
    assert len(SUPPORTED) <= ci_versions.MAX_MATRIX_ROWS


def test_ci_matrix_equals_the_derived_window():
    assert _ci_matrix_minors() == SUPPORTED, (
        "ci.yml matrix disagrees with scripts/ci_versions.py; run "
        "`python scripts/ci_versions.py` and paste its output into the matrix"
    )


def _matrix_comment_block() -> str:
    """Return the comment lines documenting the ``python: [...]`` matrix entry."""
    lines = CI_WORKFLOW_PATH.read_text(encoding="utf-8").splitlines()
    index = next(
        position for position, line in enumerate(lines) if re.match(r"^\s+python:\s*\[", line)
    )
    start = index
    while start > 0:
        previous = lines[start - 1]
        if (
            previous.strip().startswith("#")
            or not previous.strip()
            or re.match(r"^\s*matrix:", previous)
        ):
            start -= 1
            continue
        break
    return "\n".join(line for line in lines[start:index] if line.strip().startswith("#"))


def test_ci_matrix_documents_where_the_list_comes_from():
    comment = _matrix_comment_block()
    assert comment, "the matrix entry carries no explanatory comment"
    assert "scripts/ci_versions.py" in comment, (
        "the matrix must point at scripts/ci_versions.py so the next edit goes "
        "to the source of truth, not to a copy"
    )


def test_ceiling_reason_is_recorded():
    reason = ci_versions.CEILING_REASON.strip()
    assert len(reason) > 60, "the ceiling needs a real reason, not a stub"
    assert "TODO" not in reason
    assert re.search(r"[a-z0-9][\w.-]*\d", reason), "the reason must name the concrete blocker"
    assert ci_versions.NEWEST_SUPPORTED_MINOR[0] == 3


def test_distribution_metadata_declares_the_supported_floor():
    declared = _pyproject()["project"]["requires-python"]
    package_metadata = metadata("fastretrieval")
    assert package_metadata["Requires-Python"] == declared
    classifiers = set(package_metadata.get_all("Classifier") or [])
    declared_classifiers = {
        item.removeprefix("Programming Language :: Python :: ")
        for item in classifiers
        if re.fullmatch(r"Programming Language :: Python :: 3\.\d+", item)
    }
    assert declared_classifiers == set(SUPPORTED)
    requirements = package_metadata.get_all("Requires-Dist") or []
    assert not any('python_version == "3.10"' in item for item in requirements)


def test_ruff_target_version_matches_the_floor():
    floor, _ = ci_versions.parse_window(ci_versions.read_requires_python())
    configured = _pyproject()["tool"]["ruff"]["target-version"]
    assert configured == f"py3{floor}", (
        f"ruff target-version {configured!r} must follow the requires-python floor py3{floor}"
    )


def test_runtime_is_inside_supported_cpython_window():
    assert sys.implementation.name == "cpython"
    floor, ceiling = SUPPORTED[0], SUPPORTED[-1]
    floor_minor = int(floor.split(".")[1])
    ceiling_minor = int(ceiling.split(".")[1])
    assert (3, floor_minor) <= sys.version_info[:2] <= (3, ceiling_minor)


def test_parse_window_rejects_unknown_clauses():
    with pytest.raises(ValueError, match="unsupported requires-python clause"):
        ci_versions.parse_window("~=3.11")
    with pytest.raises(ValueError, match="no lower bound"):
        ci_versions.parse_window("<3.15")


def test_parse_window_honours_upper_bounds():
    assert ci_versions.parse_window(">=3.11,<3.15") == (11, 15)
    assert ci_versions.parse_window(">=3.11,<=3.13") == (11, 14)
    assert ci_versions.parse_window(">3.10") == (11, None)


def _runtime_requirements(name: str) -> list[Requirement]:
    declared = [Requirement(item) for item in _pyproject()["project"]["dependencies"]]
    return [item for item in declared if item.name == name]


def _applies_on(requirement: Requirement, minor: str) -> bool:
    if requirement.marker is None:
        return True
    return requirement.marker.evaluate(
        {"python_version": minor, "python_full_version": f"{minor}.0"}
    )


def test_onnxruntime_is_declared_exactly_where_a_wheel_exists():
    # onnxruntime ships cp311..cp314 wheels and nothing for cp315 (no sdist either),
    # so declaring it on 3.15 makes the install unsatisfiable. The ONNX backend
    # turns its absence into an ImportError at model load (test_onnxruntime_optional).
    (requirement,) = _runtime_requirements("onnxruntime")
    without_wheel = {"3.15"}
    for minor in SUPPORTED:
        assert _applies_on(requirement, minor) is (minor not in without_wheel), minor


def test_every_supported_minor_gets_exactly_one_numpy_line():
    requirements = _runtime_requirements("numpy")
    for minor in SUPPORTED:
        applicable = [item for item in requirements if _applies_on(item, minor)]
        assert len(applicable) == 1, (minor, [str(item) for item in applicable])
