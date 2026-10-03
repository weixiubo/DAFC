"""Append-only attempts and the single predecision retry rule."""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
from typing import Any

from dafc.storage import load_json


def result_path(root: Path, actor: str, job: dict[str, Any], attempt: int) -> Path:
    return root / "results" / actor / f"{job['job_id']}--a{attempt:02d}.json"


def append_ledger(root: Path, actor: str, row: dict[str, Any]) -> None:
    path = root / "ledger" / f"{actor}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC, 384)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.write(descriptor, (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def existing_attempts(root: Path, actor: str, job: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for attempt in (0, 1):
        path = result_path(root, actor, job, attempt)
        if path.exists():
            payload = load_json(path)
            rows.append(payload)
    return rows


def next_attempt(root: Path, actor: str, job: dict[str, Any]) -> int | None:
    rows = existing_attempts(root, actor, job)
    if any((row.get("outcome_bearing") is True for row in rows)):
        return None
    if not rows:
        return 0
    if len(rows) == 1 and rows[0].get("retry_permitted") is True:
        return 1
    return None
