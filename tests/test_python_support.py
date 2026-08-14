"""Contract checks for the supported CPython versions."""

import sys
from importlib.metadata import metadata


def test_runtime_is_inside_supported_cpython_window():
    assert sys.implementation.name == "cpython"
    assert (3, 10) <= sys.version_info[:2] <= (3, 14)


def test_distribution_metadata_declares_supported_floor():
    package_metadata = metadata("fastretrieval")
    assert package_metadata["Requires-Python"] == ">=3.10"
    classifiers = set(package_metadata.get_all("Classifier") or [])
    for minor in range(10, 15):
        assert f"Programming Language :: Python :: 3.{minor}" in classifiers
