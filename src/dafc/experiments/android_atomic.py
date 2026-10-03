from __future__ import annotations

from pathlib import Path
from typing import Any

from dafc.adapters.android.device import AndroidDevice
from dafc.experiments.atomic_design import jobs_for_actor as jobs_for_actor
from dafc.experiments.atomic_state import TrialConfig, TrialState
from dafc.experiments.atomic_steps import (
    cleanup_trial,
    execute_trial_action,
    failure_payload,
    observe_trial,
    prepare_trial,
    resolve_trial_action,
    successful_payload,
)
from dafc.experiments.attempts import append_ledger
from dafc.experiments.attempts import next_attempt as next_attempt
from dafc.routing import STANDARD_MIN_INLIERS as STANDARD_MIN_INLIERS
from dafc.routing import standard_registration as standard_registration
from dafc.storage import sha256, write_new_json


def run_one(
    *,
    env: Any,
    task_registry: dict[str, Any],
    manifest: dict[str, Any],
    manifest_path: Path,
    actor: str,
    service_url: str,
    root: Path,
    job: dict[str, Any],
    attempt: int,
    device: AndroidDevice,
) -> dict[str, Any]:
    config = TrialConfig(
        env=env,
        task_registry=task_registry,
        manifest=manifest,
        manifest_path=manifest_path,
        actor=actor,
        service_url=service_url,
        root=root,
        job=job,
        attempt=attempt,
        device=device,
    )
    state = TrialState()
    prepare_trial(config, state)
    try:
        observe_trial(config, state)
        resolve_trial_action(config, state)
        execute_trial_action(config, state)
        payload = successful_payload(config, state)
    except Exception as caught:
        payload = failure_payload(config, state, caught)
    finally:
        cleanup_trial(config, state)
    write_new_json(state.path, payload)
    append_ledger(
        root,
        actor,
        {
            "attempt_path": str(state.path.resolve()),
            "attempt_sha256": sha256(state.path),
            "job_id": job["job_id"],
            "attempt": attempt,
            "status": payload["status"],
            "outcome_bearing": payload["outcome_bearing"],
            "retry_permitted": payload["retry_permitted"],
        },
    )
    return payload
