"""Tests for common type definitions."""

import typing
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fastretrieval.common.types import Device, NumpyArray, OnnxProvider, PathInput


def test_device_enum() -> None:
    """Test Device enum values."""
    assert Device.CPU == "cpu"
    assert Device.CUDA == "cuda"
    assert Device.AUTO == "auto"

    assert Device.CPU.value == "cpu"
    assert Device.CUDA.value == "cuda"
    assert Device.AUTO.value == "auto"

    assert list(Device) == [Device.CPU, Device.CUDA, Device.AUTO]


def test_path_input_type() -> None:
    """Test PathInput type definition."""
    # We can inspect the args of the union type
    args = typing.get_args(PathInput)
    assert str in args
    assert Path in args
    assert len(args) == 2


def test_onnx_provider_type() -> None:
    """Test OnnxProvider type definition."""
    args = typing.get_args(OnnxProvider)
    assert str in args
    assert tuple[str, dict[typing.Any, typing.Any]] in args
    assert len(args) == 2


def test_numpy_array_type() -> None:
    """Test NumpyArray type definition."""
    args = typing.get_args(NumpyArray)

    # Check that it contains all expected NDArray types with correct dtypes
    # numpy types compare with equality
    expected_dtypes = [
        np.float64,
        np.float32,
        np.float16,
        np.int8,
        np.int64,
        np.int32,
    ]

    # Compare whole aliases rather than digging into their parameters: numpy 2.4
    # spells NDArray[T] as ndarray[tuple[Any, ...], dtype[T]] (two parameters), but
    # numpy 2.5 -- the line installed on CPython 3.15 -- turned NDArray into a type
    # alias whose only parameter is T. Alias equality holds on both.
    assert len(args) == len(expected_dtypes)
    assert set(args) == {NDArray[dtype] for dtype in expected_dtypes}
