import importlib.util

import pytest

from fastretrieval.convert.modal_backend import BACKENDS, resolve_backend


def test_local_is_the_default():
    assert resolve_backend(None) == "local"


def test_modal_backend_is_selectable():
    assert resolve_backend("modal") == "modal"


def test_unknown_backend_lists_the_valid_ones():
    with pytest.raises(ValueError, match="local"):
        resolve_backend("ec2")


def test_modal_backend_explains_itself_when_modal_is_absent(monkeypatch):
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: None if name == "modal" else importlib.util.find_spec(name),
    )
    from fastretrieval.convert.modal_backend import run_remote

    with pytest.raises(ImportError, match="modal"):
        run_remote("onnx", source="acme/tiny", out="/tmp/out")


def test_backends_tuple_is_the_single_source_of_truth():
    assert BACKENDS == ("local", "modal")
