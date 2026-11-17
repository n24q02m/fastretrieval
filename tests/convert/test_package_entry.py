"""Package entry points: the dependency probe and ``python -m fastretrieval.convert``."""

import runpy
import sys

import pytest

from fastretrieval.convert import REQUIREMENTS, require_convert_deps


def test_require_convert_deps_accepts_importable_modules():
    assert require_convert_deps("json", "numpy") is None


def test_require_convert_deps_treats_a_missing_parent_package_as_missing():
    # find_spec raises ModuleNotFoundError (instead of returning None) for a dotted
    # name whose parent package is absent; the probe must report it, not crash.
    with pytest.raises(ImportError, match=r"dependencies: no_such_parent_pkg_xyz\.child\."):
        require_convert_deps("json", "no_such_parent_pkg_xyz.child")


def test_require_convert_deps_lists_every_missing_module_and_the_requirements_file():
    with pytest.raises(ImportError) as excinfo:
        require_convert_deps("missing_mod_a_xyz", "json", "missing_mod_b_xyz")

    message = str(excinfo.value)
    assert "Missing conversion dependencies: missing_mod_a_xyz, missing_mod_b_xyz." in message
    assert f"uv run --with-requirements {REQUIREMENTS}" in message
    assert REQUIREMENTS.name == "requirements.txt"
    assert REQUIREMENTS.is_file()


def test_module_entry_point_prints_help_and_exits_2_without_a_command(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["fastretrieval.convert"])

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("fastretrieval.convert", run_name="__main__")

    assert excinfo.value.code == 2
    assert "usage: python -m fastretrieval.convert" in capsys.readouterr().out
