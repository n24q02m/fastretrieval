"""HTTP surface: a shared ``/embed`` + ``/rerank`` model server.

The server lives in this distribution so the model runtime and the HTTP
surface that serves it have one owner. Nothing here is imported by the
library API: ``fastretrieval.server.__init__`` only exposes ``__version__``,
so the package stays importable without the optional ``[server]`` extra.
"""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("fastretrieval")
except importlib.metadata.PackageNotFoundError:  # source tree, not installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
