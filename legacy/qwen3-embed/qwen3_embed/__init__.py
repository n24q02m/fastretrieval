"""Old import name of fastretrieval, kept only so existing imports keep working."""

import importlib.metadata
import warnings

from fastretrieval import (
    CustomModelSpec,
    CustomRerankerSpec,
    Device,
    TextCrossEncoder,
    TextEmbedding,
)

warnings.warn(
    "qwen3-embed has been renamed to fastretrieval; `qwen3_embed` only re-exports "
    "fastretrieval's API and receives no further releases. Install `fastretrieval` and "
    "import from `fastretrieval` instead.",
    DeprecationWarning,
    stacklevel=2,
)

try:
    __version__ = importlib.metadata.version("qwen3-embed")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

# Exactly the public names qwen3-embed 1.14.0 exported.
__all__ = [
    "CustomModelSpec",
    "CustomRerankerSpec",
    "Device",
    "TextEmbedding",
    "TextCrossEncoder",
]
