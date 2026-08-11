"""Tests for common type definitions."""

import typing
from pathlib import Path

import numpy as np

from qwen3_embed.common.types import Device, NumpyArray, OnnxProvider, PathInput


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

    # typing.get_args will return np.ndarray[Any, np.dtype[np.float64]]
    # We can extract the actual type by looking at __args__[1].__args__[0]

    actual_dtypes = []
    for arg in args:
        # arg is NDArray[np.float64] which is numpy.ndarray[typing.Any, numpy.dtype[numpy.float64]]
        # arg.__args__[1] is numpy.dtype[numpy.float64]
        # arg.__args__[1].__args__[0] is numpy.float64
        actual_dtypes.append(arg.__args__[1].__args__[0])

    for dtype in expected_dtypes:
        assert dtype in actual_dtypes

    assert len(actual_dtypes) == len(expected_dtypes)
