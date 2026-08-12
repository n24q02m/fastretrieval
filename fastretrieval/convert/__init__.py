"""Optional model-conversion helpers with lazy heavy dependencies.

The runtime dependency set deliberately does not include the conversion stack.
Run conversion in a throwaway environment with the requirements file bundled
next to this module::

    uv run --with-requirements fastretrieval/convert/requirements.txt \\
      python -m fastretrieval.convert onnx <model-id> --out ./out
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

REQUIREMENTS = Path(__file__).with_name("requirements.txt")

_HOWTO = (
    "Missing conversion dependencies: {missing}.\n"
    "They are deliberately not declared in pyproject (see the module docstring).\n"
    "Run the converter in a throwaway environment instead:\n"
    "  uv run --with-requirements {req} python -m fastretrieval.convert ..."
)


def require_convert_deps(*modules: str) -> None:
    """Raise an actionable error when conversion modules are unavailable."""
    missing: list[str] = []
    for module in modules:
        try:
            available = importlib.util.find_spec(module) is not None
        except ModuleNotFoundError:
            available = False
        if not available:
            missing.append(module)
    if missing:
        raise ImportError(_HOWTO.format(missing=", ".join(missing), req=REQUIREMENTS))


__all__ = ["REQUIREMENTS", "require_convert_deps"]
