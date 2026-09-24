#!/usr/bin/env python3
"""参赛程序入口（与 Demo 同款）：python main3.py <port>"""
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _setup_logging(root: Path) -> None:
    logs = root / "logs"
    logs.mkdir(exist_ok=True)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    file_handler = RotatingFileHandler(
        logs / "agent.log", maxBytes=8 * 1024 * 1024, backupCount=2, encoding="utf-8",
    )
    file_handler.setFormatter(fmt)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(fmt)
    logging.basicConfig(level=logging.INFO, handlers=[file_handler, stream_handler])


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python main3.py <port>")
    port = int(sys.argv[1])
    root = Path(__file__).resolve().parent
    os.chdir(root)
    sys.path.insert(0, str(root / "src"))

    _setup_logging(root)

    from agent.server import serve

    logging.getLogger(__name__).info("listening on 0.0.0.0:%d", port)
    serve(port)


if __name__ == "__main__":
    main()
