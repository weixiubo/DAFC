#!/usr/bin/env python3
"""Shared protocol primitives for the TransitionGuard formal experiment."""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


PROTOCOL_VERSION = "transitionguard-aaai27-formal-v1"
SOURCE_CANDIDATE_CAP = 128
SOURCE_SUCCESSES_PER_TASK = 12
MINIMUM_DISTRACTOR_TRAJECTORIES = 7


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json_sha256(row: Any) -> str:
    payload = json.dumps(
        row, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(path: Path, row: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"refusing to replace existing artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    if temporary.exists():
        raise FileExistsError(f"stale atomic-write temporary exists: {temporary}")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(row, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    # A same-directory hard link publishes the fully flushed inode while
    # preserving O_EXCL semantics for the final name.  Path.replace() would
    # silently overwrite an existing formal artifact after a resume race.
    os.link(temporary, path)
    temporary.unlink()


def verify_external_file_bindings(rows: Iterable[dict[str, Any]]) -> None:
    """Verify absolute, hash-pinned runtime files declared by a manifest."""

    for row in rows:
        path = Path(str(row["path"]))
        if not path.is_absolute() or not path.is_file():
            raise RuntimeError(f"bound external file is unavailable: {path}")
        observed_bytes = path.stat().st_size
        if observed_bytes != int(row["bytes"]):
            raise RuntimeError(f"bound external file size changed: {path}")
        if sha256(path) != str(row["sha256"]):
            raise RuntimeError(f"bound external file hash changed: {path}")


def directory_fingerprint(path: Path) -> dict[str, Any]:
    """Hash a runtime package tree while ignoring interpreter bytecode."""

    root = path.resolve()
    if not root.is_dir():
        raise RuntimeError(f"runtime package tree is unavailable: {root}")
    rows = []
    for child in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = child.relative_to(root)
        if "__pycache__" in relative.parts or child.suffix in {".pyc", ".pyo"}:
            continue
        rows.append(
            {
                "path": relative.as_posix(),
                "bytes": child.stat().st_size,
                "sha256": sha256(child),
            }
        )
    return {
        "files": len(rows),
        "bytes": sum(int(row["bytes"]) for row in rows),
        "sha256": canonical_json_sha256(rows),
    }


def verify_external_tree_bindings(rows: Iterable[dict[str, Any]]) -> None:
    """Verify tree-level hashes for installed runtime packages."""

    for row in rows:
        path = Path(str(row["path"]))
        if not path.is_absolute():
            raise RuntimeError(f"bound external tree is not absolute: {path}")
        observed = directory_fingerprint(path)
        expected = {
            "files": int(row["files"]),
            "bytes": int(row["bytes"]),
            "sha256": str(row["sha256"]),
        }
        if observed != expected:
            raise RuntimeError(f"bound external tree changed: {path}")


def verify_external_git_bindings(rows: Iterable[dict[str, Any]]) -> None:
    """Verify external source repositories used by the formal runtime."""

    for row in rows:
        path = Path(str(row["path"]))
        if not path.is_absolute() or not (path / ".git").exists():
            raise RuntimeError(f"bound external repository is unavailable: {path}")
        head = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.strip()
        if head != str(row["commit"]):
            raise RuntimeError(f"bound external repository changed: {path}")
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout.splitlines()
        expected_status = row.get("status_porcelain")
        if expected_status is not None and status != expected_status:
            raise RuntimeError(f"bound external repository status changed: {path}")
        for binding in row.get("status_file_bindings", []):
            changed = path / str(binding["path"])
            if not changed.is_file():
                raise RuntimeError(
                    f"bound external repository file is unavailable: {changed}"
                )
            if changed.stat().st_size != int(binding["bytes"]):
                raise RuntimeError(
                    f"bound external repository file size changed: {changed}"
                )
            if sha256(changed) != str(binding["sha256"]):
                raise RuntimeError(
                    f"bound external repository file hash changed: {changed}"
                )
        if row.get("require_clean") and status:
            raise RuntimeError(f"bound external repository is dirty: {path}")


def walk_seed_fields(value: Any) -> Iterable[int]:
    """Yield integer ``seed`` fields below a task-keyed JSON subtree."""

    if isinstance(value, dict):
        seed = value.get("seed")
        if isinstance(seed, int) and not isinstance(seed, bool):
            yield seed
        for child in value.values():
            yield from walk_seed_fields(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_seed_fields(child)


def walk_task_seeds(value: Any, tasks: set[str]) -> Iterable[tuple[str, int]]:
    """Yield task/seed pairs in both row-wise and task-keyed JSON schemas.

    Development artifacts use two common layouts: records containing explicit
    ``task`` and ``seed`` fields, and mappings such as
    ``{"click-option": [{"seed": ...}]}``.  Formal candidate freshness must
    conservatively cover both; otherwise merely previewed protocol seeds could
    be selected again after their values were observed.
    """

    if isinstance(value, dict):
        task = value.get("task")
        seed = value.get("seed")
        if (
            isinstance(task, str)
            and task in tasks
            and isinstance(seed, int)
            and not isinstance(seed, bool)
        ):
            yield task, seed
        for key, child in value.items():
            if key in tasks:
                for nested_seed in walk_seed_fields(child):
                    yield key, nested_seed
            yield from walk_task_seeds(child, tasks)
    elif isinstance(value, list):
        for child in value:
            yield from walk_task_seeds(child, tasks)


def development_seed_inventory(
    validation_dir: Path, tasks: Iterable[str]
) -> tuple[dict[str, set[int]], dict[str, str]]:
    task_set = set(tasks)
    result = {task: set() for task in task_set}
    bindings: dict[str, str] = {}
    for path in sorted(validation_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        found = list(walk_task_seeds(payload, task_set))
        if not found:
            continue
        bindings[path.name] = sha256(path)
        for task, seed in found:
            result[task].add(seed)
    return result, bindings


def candidate_seed(task: str, index: int) -> int:
    if not task or index < 0:
        raise ValueError("candidate seed requires task and non-negative index")
    digest = hashlib.sha256(
        f"{PROTOCOL_VERSION}|source|{task}|{index}".encode("utf-8")
    ).digest()
    seed = int.from_bytes(digest[:8], "big") & 0x7FFFFFFF
    return seed or 1


def eligible_candidate_seeds(
    task: str, excluded: Iterable[int], count: int = SOURCE_CANDIDATE_CAP
) -> list[dict[str, int]]:
    if count <= 0:
        raise ValueError("candidate count must be positive")
    blocked = set(int(row) for row in excluded)
    rows = []
    seen = set()
    index = 0
    while len(rows) < count:
        seed = candidate_seed(task, index)
        if seed not in blocked and seed not in seen:
            rows.append({"derivation_index": index, "seed": seed})
            seen.add(seed)
        index += 1
        if index > count * 100:
            raise RuntimeError("could not derive enough unique formal seeds")
    return rows


@dataclass(frozen=True)
class ExperienceRecord:
    record_id: str
    goal: str
    task: str
    seed: int
    source_episode: str

    def __post_init__(self) -> None:
        if not self.record_id or not self.goal or not self.task:
            raise ValueError("experience identity fields must be non-empty")
        if not isinstance(self.seed, int) or isinstance(self.seed, bool):
            raise ValueError("experience seed must be an integer")
        if not self.source_episode:
            raise ValueError("experience source path must be non-empty")


class ExperienceIndex:
    def __init__(self, records: Iterable[ExperienceRecord]):
        self.records = tuple(records)
        ids = [row.record_id for row in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("experience record IDs must be unique")

    def retrieve_exact(
        self, goal: str, *, minimum_distractors: int = MINIMUM_DISTRACTOR_TRAJECTORIES
    ) -> tuple[list[ExperienceRecord], dict[str, Any]]:
        matched = [row for row in self.records if row.goal == goal]
        rejected = [row for row in self.records if row.goal != goal]
        if len(rejected) < minimum_distractors:
            raise RuntimeError("experience query lacks required distractors")
        return matched, {
            "retrieval": "exact normalized goal equality",
            "query_goal": goal,
            "bank_records": len(self.records),
            "matched_records": len(matched),
            "rejected_records": len(rejected),
            "matched_record_ids": [row.record_id for row in matched],
            "rejected_record_ids": [row.record_id for row in rejected],
            "minimum_distractors": minimum_distractors,
            "step_or_progress_input": None,
        }


class AppendOnlyLedger:
    REQUIRED = {"record_id", "artifact", "artifact_sha256", "status"}

    def __init__(self, path: Path, root: Path):
        self.path = path
        self.root = root.resolve()

    def artifact_path(self, relative: str) -> Path:
        candidate = (self.root / relative).resolve()
        if candidate == self.root or self.root not in candidate.parents:
            raise RuntimeError("ledger artifact escapes its declared root")
        return candidate

    def read(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        with self.path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                row = json.loads(line)
                missing = self.REQUIRED - set(row)
                if missing:
                    raise RuntimeError(
                        f"ledger line {line_number} lacks fields: {sorted(missing)}"
                    )
                rows.append(row)
        ids = [str(row["record_id"]) for row in rows]
        if len(ids) != len(set(ids)):
            raise RuntimeError("ledger contains duplicate record IDs")
        return rows

    def verify(self) -> list[dict[str, Any]]:
        rows = self.read()
        for row in rows:
            artifact = self.artifact_path(str(row["artifact"]))
            if not artifact.is_file():
                raise RuntimeError(f"ledger artifact is missing: {artifact}")
            if sha256(artifact) != row["artifact_sha256"]:
                raise RuntimeError(f"ledger artifact hash mismatch: {artifact}")
        return rows

    def append(self, row: dict[str, Any]) -> None:
        missing = self.REQUIRED - set(row)
        if missing:
            raise ValueError(f"ledger row lacks fields: {sorted(missing)}")
        existing = self.verify()
        if any(old["record_id"] == row["record_id"] for old in existing):
            raise FileExistsError(f"ledger record already exists: {row['record_id']}")
        artifact = self.artifact_path(str(row["artifact"]))
        if not artifact.is_file() or sha256(artifact) != row["artifact_sha256"]:
            raise RuntimeError("new ledger artifact is absent or hash-mismatched")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())


class LedgerSession:
    """Verify a ledger once, then append in O(1) within one writer process."""

    def __init__(self, ledger: AppendOnlyLedger):
        self.ledger = ledger
        self.rows = ledger.verify()
        self.record_ids = {str(row["record_id"]) for row in self.rows}

    def append(self, row: dict[str, Any]) -> None:
        missing = self.ledger.REQUIRED - set(row)
        if missing:
            raise ValueError(f"ledger row lacks fields: {sorted(missing)}")
        record_id = str(row["record_id"])
        if record_id in self.record_ids:
            raise FileExistsError(f"ledger record already exists: {record_id}")
        artifact = self.ledger.artifact_path(str(row["artifact"]))
        if (
            not artifact.is_file()
            or sha256(artifact) != row["artifact_sha256"]
        ):
            raise RuntimeError("new ledger artifact is absent or hash-mismatched")
        self.ledger.path.parent.mkdir(parents=True, exist_ok=True)
        with self.ledger.path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        self.rows.append(row)
        self.record_ids.add(record_id)
