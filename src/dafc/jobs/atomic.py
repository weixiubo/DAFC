"""Fixed-design atomic batches with immutable attempts and conservative recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from dafc.configuration import RunConfig
from dafc.evaluation.attempts import sequence_errors
from dafc.experiments.android_atomic import run_one
from dafc.experiments.atomic_design import TASK_BY_ID, TASKS, jobs_for_actor
from dafc.experiments.atomic_inputs import validate_source
from dafc.experiments.attempts import existing_attempts, next_attempt, result_path
from dafc.resources import sha256_file
from dafc.storage import canonical_sha256, write_new_json


def prepare_atomic(config: RunConfig) -> dict[str, Any]:
    source_path = config.inputs["source_registry"].path
    validated = _source_registry(config, source_path)
    actor = config.parameters["actor"]
    jobs = jobs_for_actor(actor)
    limit = config.parameters.get("limit", len(jobs))
    if isinstance(limit, bool) or not isinstance(limit, int) or (not 1 <= limit <= len(jobs)):
        raise ValueError("limit must be an integer from 1 to 560")
    scheduled = [
        {
            **job,
            "design_job_id": job["job_id"],
            "job_id": config.run_id + "--" + hashlib.sha256(job["job_id"].encode()).hexdigest()[:24],
        }
        for job in jobs[:limit]
    ]
    design = {
        "tasks": [{"task_id": task.task_id, "source": validated[task.task_id]} for task in TASKS],
        "planned_jobs": {actor: scheduled},
    }
    return {
        "manifest": {
            "identity": "dafc-configured-atomic-v1",
            "paper_evidence": False,
            "partial_design": limit < len(jobs),
            "design": design,
            "jobs_sha256": canonical_sha256(design["planned_jobs"]),
        }
    }


def _verify_attempt_images(attempt: dict[str, Any], job: dict[str, Any], manifest_path: Path) -> None:
    if attempt.get("job") != job or attempt.get("manifest_sha256") != sha256_file(manifest_path):
        raise ValueError("Atomic attempt design binding differs from experiment manifest")
    for key in ("before", "before_raw", "after"):
        image = attempt.get(key)
        if image is not None and sha256_file(Path(image["path"])) != image["sha256"]:
            raise ValueError(f"Atomic attempt {key} image checksum mismatch")


def _ledger_bindings(root: Path, actor: str, jobs: list[dict[str, Any]]) -> set[str]:
    ledger = root / "ledger" / f"{actor}.jsonl"
    entries = (
        [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
        if ledger.exists()
        else []
    )
    expected = {
        str(result_path(root, actor, job, attempt).resolve()): (job["job_id"], attempt)
        for job in jobs
        for attempt in (0, 1)
    }
    seen: set[str] = set()
    for entry in entries:
        value = entry["attempt_path"]
        if value not in expected or value in seen:
            raise ValueError("Unexpected or duplicate atomic ledger entry")
        if (entry.get("job_id"), entry.get("attempt")) != expected[value]:
            raise ValueError("Atomic ledger job identity mismatch")
        seen.add(value)
        if sha256_file(Path(value)) != entry["attempt_sha256"]:
            raise ValueError("Atomic attempt hash differs from ledger")
    return seen


def _check_inflight(root: Path, actor: str, jobs: list[dict[str, Any]], seen: set[str]) -> None:
    marker = root / "inflight.json"
    if not marker.exists():
        return
    inflight = json.loads(marker.read_text())
    if inflight["job"] not in jobs or inflight["attempt"] not in (0, 1):
        raise ValueError("In-flight record differs from scheduled design")
    path = result_path(root, actor, inflight["job"], inflight["attempt"])
    if str(path.resolve()) not in seen:
        raise ValueError(
            "Interrupted trial may have executed an action without a durable outcome; inspect it before starting a new run"
        )
    marker.unlink()


def verify_recovery(root: Path, actor: str, jobs: list[dict[str, Any]]) -> None:
    seen = _ledger_bindings(root, actor, jobs)
    for job in jobs:
        attempts = existing_attempts(root, actor, job)
        errors = sequence_errors(attempts, require_complete=False)
        if errors:
            raise ValueError("; ".join(errors))
        for attempt in attempts:
            path = result_path(root, actor, job, attempt["attempt"])
            if str(path.resolve()) not in seen:
                raise ValueError("Atomic attempt has no durable ledger binding")
            _verify_attempt_images(attempt, job, root / "experiment.json")
    _check_inflight(root, actor, jobs, seen)


def run_batch(config: RunConfig, prepared: dict[str, Any]) -> dict[str, Any]:
    from dafc.jobs.platforms import android_session

    manifest = prepared["manifest"]
    path = config.output / "experiment.json"
    if path.exists():
        if json.loads(path.read_text()) != manifest:
            raise ValueError("Atomic design differs from existing experiment manifest")
    else:
        write_new_json(path, manifest)
    actor = config.parameters["actor"]
    jobs = manifest["design"]["planned_jobs"][actor]
    verify_recovery(config.output, actor, jobs)
    with android_session(config) as (env, tasks, device):
        for job in jobs:
            attempt = next_attempt(config.output, actor, job)
            while attempt is not None:
                marker = config.output / "inflight.json"
                write_new_json(marker, {"job": job, "attempt": attempt})
                run_one(
                    env=env,
                    task_registry=tasks,
                    device=device,
                    manifest=manifest,
                    manifest_path=path,
                    actor=actor,
                    service_url=config.services["model"],
                    root=config.output,
                    job=job,
                    attempt=attempt,
                )
                marker.unlink()
                attempt = next_attempt(config.output, actor, job)
    outcomes = [
        row
        for job in jobs
        for row in existing_attempts(config.output, actor, job)
        if row.get("outcome_bearing") is True
    ]
    return {
        "workflow": config.workflow,
        "run_id": config.run_id,
        "requested": len(jobs),
        "outcomes": len(outcomes),
        "successes": sum((row.get("success") is True for row in outcomes)),
        "paper_evidence": False,
        "partial_design": manifest["partial_design"],
        "status": "completed" if len(outcomes) == len(jobs) else "incomplete",
    }


def _source_registry(config: RunConfig, source_path: Path) -> dict[str, Any]:
    registry = json.loads(source_path.read_text())
    if not isinstance(registry, dict) or not all(
        (isinstance(registry.get(name), list) for name in ("selected_successes", "retained_failures"))
    ):
        raise ValueError("Source registry requires selected_successes and retained_failures lists")
    successes = {row["task"]: {**row, "source_success": True} for row in registry["selected_successes"]}
    failures = {row["task"]: {**row, "source_success": False} for row in registry["retained_failures"]}
    if len(successes) != len(registry["selected_successes"]) or len(failures) != len(
        registry["retained_failures"]
    ):
        raise ValueError("Source registry contains duplicate tasks")
    if successes.keys() & failures.keys() or set(successes | failures) != set(TASK_BY_ID):
        raise ValueError("Source registry must contain exactly the sealed seven tasks, without overlap")
    base = (
        config.resources.expand(config.parameters["resource_base"], base=config.config_file.parent)
        if "resource_base" in config.parameters
        else source_path.parent
    )
    validated = {
        name: validate_source(row, resource_base=base) for name, row in (successes | failures).items()
    }
    return validated
