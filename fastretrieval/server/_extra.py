"""Import guard for the optional ``[server]`` extra of ``fastretrieval``.

``fastretrieval.server`` ships inside the ``fastretrieval`` wheel, but its
HTTP stack is only installed by ``pip install "fastretrieval[server]"``. The
API module and the CLI call ``require_server_extra()`` before touching that
stack, so a base install fails with the install hint instead of a bare
``ModuleNotFoundError`` for ``fastapi``. The package ``__init__`` does not
call it, so ``__version__`` stays importable without the extra.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import Any

# Import names of the [server] extra in pyproject.toml, in declaration order.
SERVER_EXTRA_MODULES: tuple[str, ...] = (
    "fastapi",
    "uvicorn",
    "pydantic",
)

INSTALL_HINT = 'pip install "fastretrieval[server]"'

FindSpec = Callable[[str], Any]


def missing_server_modules(find_spec: FindSpec = importlib.util.find_spec) -> list[str]:
    """Return the [server] extra modules that cannot be found, in declaration order."""
    return [name for name in SERVER_EXTRA_MODULES if find_spec(name) is None]


def require_server_extra(find_spec: FindSpec = importlib.util.find_spec) -> None:
    """Raise ``ImportError`` naming the missing modules when the extra is absent."""
    missing = missing_server_modules(find_spec)
    if missing:
        raise ImportError(
            f"fastretrieval.server requires the optional 'server' extra of fastretrieval; missing module(s): "
            f"{', '.join(missing)}. Install it with: {INSTALL_HINT}",
            name=missing[0],
        )
