"""CLI entry point for ``fastretrieval-server`` (``python -m fastretrieval.server``)."""

from __future__ import annotations

import argparse
import os
import sys

from fastretrieval.common.utils import CACHE_PATH_ENV
from fastretrieval.server._extra import require_server_extra
from fastretrieval.server.registry import THREADS_ENV


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="fastretrieval-server",
        description="Shared embed/rerank model server for the fastretrieval runtime",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=9800,
        help="Bind port (default: 9800)",
    )
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        help="uvicorn log level (default: info)",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        metavar="DIR",
        help=f"Model cache directory (env {CACHE_PATH_ENV}; default: fastretrieval user cache)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help=f"ONNX intra-op threads per session (env {THREADS_ENV})",
    )
    args = parser.parse_args()

    if args.cache_dir:
        os.environ[CACHE_PATH_ENV] = args.cache_dir
    if args.threads and args.threads > 0:
        os.environ[THREADS_ENV] = str(args.threads)

    try:
        require_server_extra()
    except ImportError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    import uvicorn

    uvicorn.run(
        "fastretrieval.server.api:app",
        host=args.host,
        port=args.port,
        log_level=args.log_level,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
