"""Compatibility shims for the oldest supported CPython runtime."""

from enum import Enum


class StrEnum(str, Enum):
    """Backport the small ``enum.StrEnum`` surface used by the package."""

    def __str__(self) -> str:
        return str(self.value)
