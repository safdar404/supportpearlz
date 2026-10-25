"""Central logging: console + rotating file, plus a JSONL turn log for evaluation and audits."""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.config import scrub

_done = False


def setup_logging(level: str = "INFO", log_dir: Path | None = None) -> logging.Logger:
    global _done
    log = logging.getLogger("supportpearlz")
    if _done:
        return log
    log.setLevel(getattr(logging, level.upper(), logging.INFO))
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    log.addHandler(ch)
    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        fh = RotatingFileHandler(Path(log_dir) / "app.log", maxBytes=2_000_000, backupCount=3, encoding="utf-8")
        fh.setFormatter(fmt)
        log.addHandler(fh)
    log.propagate = False
    _done = True
    return log


def log_turn(log_dir: Path, record: dict) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"), **record}
    with (Path(log_dir) / "turns.jsonl").open("a", encoding="utf-8") as f:
        f.write(scrub(json.dumps(record, ensure_ascii=False, default=str)) + "\n")


def read_turns(log_dir: Path, limit: int = 200) -> list[dict]:
    p = Path(log_dir) / "turns.jsonl"
    if not p.exists():
        return []
    lines = p.read_text(encoding="utf-8").splitlines()[-limit:]
    return [json.loads(l) for l in lines if l.strip()]
