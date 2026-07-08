"""Entry point for the Live Client recorder.

Logs to data/recorder.log (rotating, always on) and additionally to the
console when run with a visible terminal. Under pythonw.exe (no console)
the file log is the only output.

Usage (manual):
    .venv\\Scripts\\python.exe scripts\\record.py

Usage (silent background):
    .venv\\Scripts\\pythonw.exe scripts\\record.py
    or double-click scripts\\record_silent.bat
"""
from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lol_coach.config import load_config
from lol_coach.recorder import run_forever


def _configure_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )

    handlers: list[logging.Handler] = []

    file_handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(fmt)
    handlers.append(file_handler)

    has_console = (
        sys.stderr is not None
        and hasattr(sys.stderr, "isatty")
        and sys.stderr.isatty()
    )
    if has_console:
        console = logging.StreamHandler()
        console.setFormatter(fmt)
        handlers.append(console)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    for h in handlers:
        root.addHandler(h)


def main() -> int:
    cfg = load_config()
    raw_dir = Path(cfg["paths"]["data_raw"])
    log_path = raw_dir.parent / "recorder.log"
    _configure_logging(log_path)

    interval = float(cfg.get("recorder", {}).get("poll_interval_s", 1.0))
    logging.info(
        "Recorder starting. Output=%s log=%s poll=%.1fs",
        raw_dir,
        log_path,
        interval,
    )
    try:
        run_forever(raw_dir, poll_interval_s=interval)
    except KeyboardInterrupt:
        logging.info("Stopped by user (Ctrl+C).")
    except Exception:
        logging.exception("Recorder crashed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
