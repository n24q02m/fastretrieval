"""``fastretrieval.server`` is part of this dist, so its version is the dist version."""

from __future__ import annotations

import importlib
import importlib.metadata
from unittest.mock import patch

import fastretrieval
import fastretrieval.server


def test_version_exposed() -> None:
    assert isinstance(fastretrieval.server.__version__, str)
    assert len(fastretrieval.server.__version__) > 0


def test_version_matches_the_dist() -> None:
    assert fastretrieval.server.__version__ == fastretrieval.__version__


def test_version_comes_from_fastretrieval_dist() -> None:
    """The server ships inside the fastretrieval wheel; there is no separate dist."""
    assert fastretrieval.server.__version__ == importlib.metadata.version("fastretrieval")


def test_version_fallback() -> None:
    """The package still imports when the dist is not installed (source tree)."""
    with patch("importlib.metadata.version", side_effect=importlib.metadata.PackageNotFoundError):
        importlib.reload(fastretrieval.server)
        assert fastretrieval.server.__version__ == "0.0.0+unknown"

    # Restore the real version for other tests.
    importlib.reload(fastretrieval.server)
