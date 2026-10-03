from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from dafc.adapters.android import device as C482
from dafc.adapters.android.device import AndroidDevice
from dafc.experiments.atomic_design import TASK_BY_ID, AtomicTask
from dafc.experiments.atomic_design import jobs_for_actor as jobs_for_actor
from dafc.experiments.atomic_state import TrialConfig, TrialState
from dafc.experiments.attempts import next_attempt as next_attempt
from dafc.experiments.attempts import result_path
from dafc.models import client as GROUND_CLIENT
from dafc.models.responses import compact_model_response, normalized_to_pixel, target_instruction
from dafc.routing import STANDARD_MIN_INLIERS as STANDARD_MIN_INLIERS
from dafc.routing import authorize, point_in_bounds
from dafc.routing import standard_registration as standard_registration
from dafc.storage import image_receipt, sha256

RUNNER_VERSION = "dafc-atomic-v4-explicit-resources"
POST_ACTION_SETTLE_S = 4.0


def execute_click(env: Any, point: list[float], image: np.ndarray) -> list[float]:
    if not point_in_bounds(point, image, 0.0):
        raise RuntimeError("target point is outside the observation")
    from android_world.env import json_action

    height, width = image.shape[:2]
    executed = [min(max(int(round(point[0])), 0), width - 1), min(max(int(round(point[1])), 0), height - 1)]
    env.execute_action(json_action.JSONAction(action_type="click", x=executed[0], y=executed[1]))
    time.sleep(POST_ACTION_SETTLE_S)
    return [float(executed[0]), float(executed[1])]


def ensure_clean_observation_ui(env: Any, stage: str, *, device: AndroidDevice) -> dict[str, Any]:
    before = device.adb("shell", "settings", "get", "system", "pointer_location")
    env.hide_automation_ui()
    time.sleep(0.25)
    after = device.adb("shell", "settings", "get", "system", "pointer_location")
    if after != "0":
        raise RuntimeError(f"pointer overlay remains enabled at {stage}: {after!r}")
    return {"stage": stage, "before": before, "after": after, "clean": True}


def setup_target_state(
    env: Any, task_registry: dict[str, Any], spec: AtomicTask, contract_id: str, *, device: AndroidDevice
):
    from android_world.env import json_action
    from android_world.env.setup_device import apps as setup_apps

    contracts = {row.contract_id: row for row in C482.CONTRACTS}
    contract = contracts[contract_id]
    device.apply_contract(contract)
    env.reset(go_home=True)
    if spec.task_id == "AudioRecorderRecordAudio":
        setup_apps.AudioRecorder.setup(env)
    task = task_registry[spec.task_id](dict(spec.params))
    try:
        task.initialize_task(env)
        ui_receipts = [ensure_clean_observation_ui(env, "after_task_initialization", device=device)]
        initial_reward = float(task.is_successful(env))
        if initial_reward != 0.0:
            raise RuntimeError("task is already successful before setup")
        env.execute_action(json_action.JSONAction(action_type="open_app", app_name=spec.app_name))
        time.sleep(6.0)
        ui_receipts.append(ensure_clean_observation_ui(env, "after_open_app", device=device))
        setup_actions = []
        for index, label in enumerate(spec.setup_prefix_labels):
            state = C482.wait_state(env, required_label=label, timeout_s=12.0)
            action = C482.click_label(env, state, label)
            reward = float(task.is_successful(env))
            setup_actions.append({"index": index, **action, "official_reward": reward})
            if reward != 0.0:
                raise RuntimeError("setup prefix completed the task before target action")
            ui_receipts.append(ensure_clean_observation_ui(env, f"after_setup_{index:02d}", device=device))
        state = C482.wait_state(env, required_label=spec.terminal_label_scoring_only, timeout_s=12.0)
        raw = C482.pixels(state)
        observed = C482.observe(raw, contract)
        reward = float(task.is_successful(env))
        if reward != 0.0:
            raise RuntimeError("target state is successful before target action")
        return (task, str(task.goal), raw, observed, setup_actions, ui_receipts)
    except BaseException:
        try:
            task.tear_down(env)
        except Exception:
            pass
        raise


def task_source(manifest: dict[str, Any], task_id: str) -> dict[str, Any]:
    rows = {row["task_id"]: row["source"] for row in manifest["design"]["tasks"]}
    return rows[task_id]


def prepare_trial(config: TrialConfig, state: TrialState):
    state.spec = TASK_BY_ID[config.job["task"]]
    state.source = task_source(config.manifest, state.spec.task_id)
    state.path = result_path(config.root, config.actor, config.job, config.attempt)
    state.image_dir = config.root / "images" / config.actor / f"{config.job['job_id']}--a{config.attempt:02d}"
    state.stage = "start"
    state.task = None
    state.model_calls = 0
    state.authorization_started = False
    state.target_action_started = False
    state.started = time.perf_counter()


def observe_trial(config: TrialConfig, state: TrialState):
    state.stage = "target_setup"
    state.task, state.goal, state.raw, state.observed, state.setup_actions, state.ui_receipts = (
        setup_target_state(
            config.env, config.task_registry, state.spec, config.job["contract"], device=config.device
        )
    )
    state.stage = "target_observation"
    state.before = image_receipt(state.image_dir / "before.png", state.observed)
    state.raw_before = image_receipt(state.image_dir / "before_raw.png", state.raw)


def resolve_trial_action(config: TrialConfig, state: TrialState):
    state.stage = "authorization"
    state.authorization_started = config.job["policy"] != "always"
    state.authorization = authorize(config.job["policy"], state.source, state.observed)
    state.model_response = None
    state.requested_action = state.authorization.get("mapped_action_xy")
    state.action_source = "source_reuse" if state.authorization["accepted"] else "target_actor"
    if not state.authorization["accepted"]:
        state.stage = "model_request"
        request_id = f"{config.job['job_id']}--a{config.attempt:02d}"
        state.model_calls = 1
        response = GROUND_CLIENT.infer(
            config.service_url,
            request_id,
            Path(state.before["path"]),
            instruction=target_instruction(state.goal),
            timeout_s=240.0,
        )
        state.model_response = compact_model_response(response)
        state.requested_action = normalized_to_pixel(
            response.get("result", {}).get("parsed_action", {}), state.observed
        )


def execute_trial_action(config: TrialConfig, state: TrialState):
    if state.requested_action is None:
        state.stage = "scientific_parse_failure"
        state.result = {
            "status": "complete",
            "scientific_failure": "target_actor_action_parse_failure",
            "success": False,
            "official_reward": 0.0,
            "action_executed": False,
            "executed_action_xy": None,
        }
        state.after = None
    else:
        state.stage = "target_action"
        state.target_action_started = True
        executed_action = execute_click(config.env, state.requested_action, state.raw)
        state.stage = "reward_read"
        state.ui_receipts.append(
            ensure_clean_observation_ui(config.env, "after_target_action", device=config.device)
        )
        next_state = config.env.get_state(wait_to_stabilize=True)
        next_raw = C482.pixels(next_state)
        next_observed = C482.observe(
            next_raw, {row.contract_id: row for row in C482.CONTRACTS}[config.job["contract"]]
        )
        state.after = image_receipt(state.image_dir / "after.png", next_observed)
        reward = float(state.task.is_successful(config.env))
        state.result = {
            "status": "complete",
            "scientific_failure": None if reward == 1.0 else "official_reward_zero",
            "success": reward == 1.0,
            "official_reward": reward,
            "action_executed": True,
            "executed_action_xy": executed_action,
        }


def successful_payload(config: TrialConfig, state: TrialState):
    payload = {
        "title": "Native-Android Atomic Closed-Loop Job",
        "paper_evidence": False,
        "runner_version": RUNNER_VERSION,
        "manifest": str(config.manifest_path.resolve()),
        "manifest_sha256": sha256(config.manifest_path),
        "jobs_sha256": config.manifest["jobs_sha256"],
        "job": config.job,
        "attempt": config.attempt,
        "source": state.source,
        "goal": state.goal,
        "setup": {
            "accessibility_scoring_only": True,
            "actions": state.setup_actions,
            "reward_remained_zero": all((row["official_reward"] == 0.0 for row in state.setup_actions)),
        },
        "observation_ui": {
            "all_clean": all((row["clean"] for row in state.ui_receipts)),
            "receipts": state.ui_receipts,
        },
        "before": state.before,
        "before_raw": state.raw_before,
        "authorization": state.authorization,
        "action_source": state.action_source,
        "model_calls": state.model_calls,
        "model_response": state.model_response,
        "requested_action_xy": state.requested_action,
        "after": state.after,
        **state.result,
        "outcome_bearing": True,
        "retry_permitted": False,
        "elapsed_s": time.perf_counter() - state.started,
    }
    return payload


def failure_payload(config: TrialConfig, state: TrialState, caught: Exception):
    outcome_bearing = bool(
        state.stage
        in {"target_observation", "authorization", "model_request", "target_action", "reward_read"}
        or state.model_calls
        or state.authorization_started
        or state.target_action_started
    )
    payload = {
        "title": "Native-Android Atomic Closed-Loop Job Failure",
        "paper_evidence": False,
        "runner_version": RUNNER_VERSION,
        "manifest": str(config.manifest_path.resolve()),
        "manifest_sha256": sha256(config.manifest_path),
        "jobs_sha256": config.manifest["jobs_sha256"],
        "job": config.job,
        "attempt": config.attempt,
        "status": "outcome_failure" if outcome_bearing else "infrastructure_failure",
        "stage": state.stage,
        "error_type": type(caught).__name__,
        "error": str(caught),
        "model_calls": state.model_calls,
        "authorization_started": state.authorization_started,
        "target_action_started": state.target_action_started,
        "outcome_bearing": outcome_bearing,
        "retry_permitted": not outcome_bearing and config.attempt == 0,
        "success": False if outcome_bearing else None,
        "elapsed_s": time.perf_counter() - state.started,
    }
    return payload


def cleanup_trial(config: TrialConfig, state: TrialState):
    if state.task is not None:
        try:
            state.task.tear_down(config.env)
        except Exception:
            pass
    try:
        config.device.restore_display()
    except Exception:
        pass
