"""Exclusive run ownership and fail-closed source/input binding for recovery."""

from __future__ import annotations

import fcntl
import hashlib
import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dafc.configuration import RunConfig
from dafc.resources import sha256_file
from dafc.storage import write_new_json


def source_identity() -> dict[str, str]:
    root = Path(__file__).parent
    files = {str(path.relative_to(root)): sha256_file(path) for path in sorted(root.rglob("*.py"))}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {"scheme": "dafc-configured-v1", "sha256": digest}


def run_binding(config: RunConfig, runtime: dict[str, Any]) -> dict[str, Any]:
    return {
        "source": source_identity(),
        "workflow": config.workflow,
        "run_id": config.run_id,
        "parameters": config.parameters,
        "services": config.services,
        "inputs": {
            name: {"path": str(item.path), "sha256": sha256_file(item.path)}
            for name, item in sorted(config.inputs.items())
        },
        "runtime": runtime,
    }


def save_status(root: Path, binding: dict[str, Any], status: str, **details: Any) -> None:
    temporary = root / ".run.json.tmp"
    temporary.write_text(json.dumps({"binding": binding, "status": status, **details}, indent=2) + "\n")
    temporary.replace(root / "run.json")


def check_resume(config: RunConfig, binding: dict[str, Any]) -> dict[str, Any] | None:
    if not config.output.exists():
        return None
    manifest = config.output / "run.json"
    if not config.resume:
        raise FileExistsError(
            f"Run directory already exists: {config.output}; choose a new run_id or enable resume"
        )
    if not manifest.is_file():
        raise ValueError("Resume manifest missing; cannot claim existing output")
    previous = json.loads(manifest.read_text())
    if previous.get("binding") != binding:
        raise ValueError("Resume source/input/config/service mismatch; use a new run_id")
    if previous.get("status") == "completed":
        result = config.output / "result.json"
        if not result.is_file() or sha256_file(result) != previous.get("result_sha256"):
            raise ValueError("Completed run result checksum mismatch")
        value = json.loads(result.read_text())
        if not isinstance(value, dict):
            raise ValueError("Completed result must be a JSON object")
        return value
    if config.workflow != "android-atomic":
        raise ValueError(
            "Interrupted live episodes cannot be safely replayed under the same ID; inspect the evidence and start a new run"
        )
    return None


@contextmanager
def own_run(config: RunConfig, binding: dict[str, Any]) -> Iterator[dict[str, Any] | None]:
    previous = check_resume(config, binding)
    if previous is not None:
        yield previous
        return
    config.output.mkdir(parents=True, exist_ok=True)
    with (config.output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        # Recheck ownership after taking the lock; no second process may start the same run.
        if (config.output / "run.json").exists():
            restored = check_resume(config, binding)
            if restored is not None:
                yield restored
                return
        save_status(config.output, binding, "running")
        try:
            yield None
        except BaseException:
            save_status(config.output, binding, "interrupted")
            raise


def finish(config: RunConfig, binding: dict[str, Any], result: dict[str, Any]) -> None:
    target = config.output / "result.json"
    write_new_json(target, result)
    save_status(config.output, binding, "completed", result_sha256=sha256_file(target))
