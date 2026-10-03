#!/usr/bin/env python3
"""Closed-loop AndroidWorld runner for real GUI models and TransitionGuard.

The runner begins after a non-point ``open_app`` bootstrap shared by every
policy.  Base actions come from the persistent audited GUI-grounder service;
replay sees only successful source transition pixels/actions and current/next
target screenshots.  Android accessibility nodes are not used for action
selection, waiting, or history.  Official task reward is read only after an
executed action for episode scoring and successful-memory admission.
"""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import argparse
import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from android_env.wrappers import a11y_grpc_wrapper
from android_world import registry
from android_world.env import env_launcher, json_action
from android_world.env.setup_device import apps as setup_apps

import dafc.adapters.android.c459_androidworld_clock_mechanics_audit as C459
import dafc.adapters.android.c482_androidworld_v2_breadth_audit as C482
import dafc.transitions as GUARD
from dafc.models.lightglue_runtime_service import LightGlueServiceClient
import dafc.models.client as GROUND_CLIENT


from dafc.paths import PROJECT
RUNNER_VERSION = "live-androidworld-transition-guard-v11"
MAX_STEPS = 20
HISTORY_HORIZON = 6
POST_ACTION_SETTLE_S = 2.0
OPEN_APP_SETTLE_S = 6.0
AUTOMATION_UI_SETTLE_S = 0.25
CONTRACTS = {row.contract_id: row for row in C482.CONTRACTS}


@dataclass(frozen=True)
class LiveTaskSpec:
    task_id: str
    params: dict[str, int | str]
    app_name: str


_LIVE_TASKS = (
    LiveTaskSpec("ClockStopWatchRunning", {}, "clock"),
    LiveTaskSpec(
        "ClockTimerEntry",
        {"hours": 0, "minutes": 1, "seconds": 30},
        "clock",
    ),
    LiveTaskSpec("SystemWifiTurnOn", {"on_or_off": "on"}, "settings"),
    LiveTaskSpec(
        "SystemBluetoothTurnOn", {"on_or_off": "on"}, "settings"
    ),
    LiveTaskSpec("SystemWifiTurnOff", {"on_or_off": "off"}, "settings"),
    LiveTaskSpec(
        "SystemBluetoothTurnOff", {"on_or_off": "off"}, "settings"
    ),
    LiveTaskSpec(
        "BrowserMaze", {"browser_task_seed": 493001}, "files"
    ),
    LiveTaskSpec("TurnOffWifiAndTurnOnBluetooth", {}, "settings"),
    LiveTaskSpec("AudioRecorderRecordAudio", {}, "audio recorder"),
)
TASKS = {row.task_id: row for row in _LIVE_TASKS}
SEQUENTIAL_POLICIES = frozenset({"raw", "naive"})
GUARDED_POLICIES = frozenset({"sg", "sgp", "full"})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture(path: Path, image: np.ndarray) -> dict[str, Any]:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not encode {path}")
    try:
        receipt_path = path.relative_to(PROJECT.resolve()).as_posix()
    except ValueError:
        # Formal artifacts are intentionally written outside the Git tree.
        # Preserve historical repository-relative receipts for development
        # runs while making the external formal image location unambiguous.
        receipt_path = path.as_posix()
    return {
        "path": receipt_path,
        "sha256": sha256(path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }


def resolve_image_receipt_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def history_text(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return "This is the task's initial in-app state."
    selected = actions[-HISTORY_HORIZON:]
    first_step = len(actions) - len(selected) + 1
    return "\n".join(
        f"Step {first_step + offset}: tapped full-screenshot pixel "
        f"coordinates ({row['action_xy'][0]:.1f}, {row['action_xy'][1]:.1f})."
        for offset, row in enumerate(selected)
    )


def instruction(goal: str, history: str) -> str:
    return (
        f"Original Android task: {goal}\n"
        "The required Android app is already open.\n"
        f"Previously executed action history:\n{history}\n"
        "Using only the current screenshot and this action history, select "
        "exactly one next executable tap that advances the original task. "
        "Do not repeat a completed action and do not explain the answer."
    )


def compact_model_response(response: dict[str, Any]) -> dict[str, Any]:
    result = response["result"]
    return {
        "service_version": response.get("service_version"),
        "kind": response.get("kind"),
        "request_sha256": response.get("request_sha256"),
        "service_s": response.get("service_s"),
        "result": {
            key: result.get(key)
            for key in (
                "id",
                "image_sha256",
                "image_width",
                "image_height",
                "input_tokens",
                "generated_tokens",
                "inference_s",
                "parsed_action",
                "raw_response",
                "instruction",
            )
        },
    }


def normalized_to_pixel(
    normalized_xy: list[float], image: dict[str, Any]
) -> tuple[tuple[float, float], bool]:
    if len(normalized_xy) != 2 or not all(
        math.isfinite(float(value)) for value in normalized_xy
    ):
        raise ValueError("model action must contain two finite coordinates")
    raw = (
        float(normalized_xy[0]) * float(image["width"]),
        float(normalized_xy[1]) * float(image["height"]),
    )
    clamped = (
        min(max(raw[0], 0.0), float(image["width"]) - 1.0),
        min(max(raw[1], 0.0), float(image["height"]) - 1.0),
    )
    return clamped, clamped != raw


def execute_click(
    env: Any, point: tuple[float, float], observation: np.ndarray
) -> tuple[float, float]:
    if not all(math.isfinite(value) for value in point):
        raise ValueError("click coordinates must be finite")
    height, width = observation.shape[:2]
    executed = (
        min(max(int(round(point[0])), 0), width - 1),
        min(max(int(round(point[1])), 0), height - 1),
    )
    env.execute_action(
        json_action.JSONAction(
            action_type="click",
            x=executed[0],
            y=executed[1],
        )
    )
    time.sleep(2.0 + POST_ACTION_SETTLE_S)
    return (float(executed[0]), float(executed[1]))


def ensure_clean_observation_ui(env: Any, stage: str) -> dict[str, Any]:
    before = C482.adb("shell", "settings", "get", "system", "pointer_location")
    env.hide_automation_ui()
    time.sleep(AUTOMATION_UI_SETTLE_S)
    after = C482.adb("shell", "settings", "get", "system", "pointer_location")
    if after != "0":
        raise RuntimeError(
            f"Android automation UI remains enabled before observation: {after!r}"
        )
    return {
        "stage": stage,
        "pointer_location_before_cleanup": before,
        "pointer_location_after_cleanup": after,
        "corrected": before != "0",
    }


def open_app(env: Any, app_name: str) -> None:
    env.execute_action(json_action.JSONAction(action_type="open_app", app_name=app_name))
    time.sleep(OPEN_APP_SETTLE_S)


def load_memory(
    path: Path,
) -> tuple[str, dict[str, int | str], list[GUARD.Transition]]:
    episode = json.loads(path.read_text(encoding="utf-8"))
    if episode.get("runner_version") != RUNNER_VERSION:
        raise RuntimeError("Android source episode runner version mismatch")
    if episode.get("policy") != "always" or episode.get("contract") != "identity":
        raise RuntimeError("Android memory must be an identity always-policy source")
    if not episode.get("success"):
        raise RuntimeError("Android memory source episode was unsuccessful")
    executed = [row for row in episode["steps"] if row["action_executed"]]
    if not executed:
        raise RuntimeError("successful Android source has no point action")
    transitions = []
    for index, row in enumerate(executed):
        before_path = resolve_image_receipt_path(row["before"]["path"])
        after_path = resolve_image_receipt_path(row["after"]["path"])
        if sha256(before_path) != row["before"]["sha256"]:
            raise RuntimeError("Android source before-image hash mismatch")
        if sha256(after_path) != row["after"]["sha256"]:
            raise RuntimeError("Android source after-image hash mismatch")
        before = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
        after = cv2.imread(str(after_path), cv2.IMREAD_COLOR)
        if before is None or after is None:
            raise RuntimeError("could not decode Android source transition")
        transitions.append(
            GUARD.Transition(
                transition_id=f"{episode['episode_id']}:transition:{index}",
                demonstration_order=index,
                source_success=True,
                before=before,
                action_xy=tuple(float(value) for value in row["action_xy"]),
                after=after,
                terminal=index == len(executed) - 1,
            )
        )
    params = episode.get("params")
    if not isinstance(params, dict):
        raise RuntimeError("Android source episode lacks task parameters")
    return str(episode["goal"]), dict(params), transitions


def load_risk_certificate(path: Path) -> GUARD.RiskCertificate:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "certificate" in payload and payload.get("pass") is not True:
        raise RuntimeError("risk-certificate qualification did not pass")
    row = payload.get("certificate", payload)
    certificate = GUARD.RiskCertificate.from_dict(row)
    if not certificate.active:
        raise RuntimeError("risk certificate is inactive")
    return certificate


def validated_timer_params(
    values: dict[str, Any] | None,
) -> dict[str, int] | None:
    if values is None:
        return None
    if set(values) != {"hours", "minutes", "seconds"}:
        raise ValueError("timer parameters must be exactly hours/minutes/seconds")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in values.values()
    ):
        raise ValueError("timer parameters must be integers")
    result = {key: int(values[key]) for key in ("hours", "minutes", "seconds")}
    if not 0 <= result["hours"] <= 23:
        raise ValueError("timer hours must be in [0, 23]")
    if not 0 <= result["minutes"] <= 59:
        raise ValueError("timer minutes must be in [0, 59]")
    if not 0 <= result["seconds"] <= 59:
        raise ValueError("timer seconds must be in [0, 59]")
    if not any(result.values()):
        raise ValueError("zero-duration timer is outside the cohort")
    return result


def validated_browser_seed(value: Any | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("browser task seed must be an integer")
    if not 0 <= value < 2**32:
        raise ValueError("browser task seed must be in [0, 2^32)")
    return int(value)


def resolved_task_spec(
    task_id: str,
    timer_params: dict[str, Any] | None,
    browser_task_seed: Any | None = None,
) -> LiveTaskSpec:
    if task_id not in TASKS:
        raise ValueError(f"unknown Android task: {task_id}")
    validated = validated_timer_params(timer_params)
    validated_seed = validated_browser_seed(browser_task_seed)
    if task_id != "ClockTimerEntry":
        if validated is not None:
            raise ValueError("timer parameters are valid only for ClockTimerEntry")
    if task_id != "BrowserMaze" and validated_seed is not None:
        raise ValueError("browser task seed is valid only for BrowserMaze")
    base = TASKS[task_id]
    if task_id == "ClockTimerEntry" and validated is not None:
        return LiveTaskSpec(
            task_id=base.task_id,
            params=validated,
            app_name=base.app_name,
        )
    if task_id == "BrowserMaze" and validated_seed is not None:
        return LiveTaskSpec(
            task_id=base.task_id,
            params={"browser_task_seed": validated_seed},
            app_name=base.app_name,
        )
    return base


def parameter_fingerprint(params: dict[str, Any]) -> str:
    encoded = json.dumps(
        params, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def sequential_replay_decision(
    policy: str,
    memory: list[GUARD.Transition],
    cursor: int,
    current: np.ndarray,
) -> tuple[dict[str, Any], tuple[float, float] | None]:
    """Select the next stored action without inspecting current pixels."""

    if policy not in SEQUENTIAL_POLICIES:
        raise ValueError(f"not a sequential replay policy: {policy}")
    ordered = sorted(memory, key=lambda row: row.demonstration_order)
    if cursor >= len(ordered):
        return {
            "method_version": "sequential-replay-baseline-v1",
            "baseline": policy,
            "decision": "fallback",
            "reason": "stored_trajectory_exhausted",
            "selected_transition_id": None,
            "selected_demonstration_order": None,
            "mapped_action_xy": None,
            "cursor": cursor,
            "pixel_content_inspected": False,
        }, None
    transition = ordered[cursor]
    if not transition.source_success:
        return {
            "method_version": "sequential-replay-baseline-v1",
            "baseline": policy,
            "decision": "fallback",
            "reason": "source_episode_not_successful",
            "selected_transition_id": transition.transition_id,
            "selected_demonstration_order": transition.demonstration_order,
            "mapped_action_xy": None,
            "cursor": cursor,
            "pixel_content_inspected": False,
        }, None
    current_height, current_width = current.shape[:2]
    source_height, source_width = transition.before.shape[:2]
    if policy == "raw":
        requested = transition.action_xy
    else:
        requested = (
            transition.action_xy[0] * current_width / source_width,
            transition.action_xy[1] * current_height / source_height,
        )
    mapped = (
        min(max(float(requested[0]), 0.0), float(current_width) - 1.0),
        min(max(float(requested[1]), 0.0), float(current_height) - 1.0),
    )
    return {
        "method_version": "sequential-replay-baseline-v1",
        "baseline": policy,
        "decision": "reuse",
        "reason": (
            "absolute_source_pixel_sequence"
            if policy == "raw"
            else "dimension_normalized_source_sequence"
        ),
        "selected_transition_id": transition.transition_id,
        "selected_demonstration_order": transition.demonstration_order,
        "source_action_xy": list(transition.action_xy),
        "requested_action_xy": list(requested),
        "mapped_action_xy": list(mapped),
        "boundary_clamped": mapped != requested,
        "source_image_shape": [source_height, source_width],
        "current_image_shape": [current_height, current_width],
        "cursor": cursor,
        "pixel_content_inspected": False,
    }, mapped


def run_episode(
    *,
    env: Any,
    task_registry: dict[str, Any],
    task_id: str,
    contract_id: str,
    policy: str,
    kind: str,
    model_service_url: str,
    run_id: str,
    attempt: int,
    output_root: Path,
    timer_params: dict[str, Any] | None = None,
    browser_task_seed: Any | None = None,
    memory_goal: str | None = None,
    memory_params: dict[str, Any] | None = None,
    memory: list[GUARD.Transition] | None = None,
    match_provider: Callable[[np.ndarray, np.ndarray], dict[str, Any]] | None = None,
    risk_certificate: GUARD.RiskCertificate | None = None,
    require_risk_certificate: bool = False,
    score_postcondition_for_ablations: bool = False,
) -> dict[str, Any]:
    if contract_id not in CONTRACTS:
        raise ValueError(f"unknown Android contract: {contract_id}")
    if policy not in {"always", *SEQUENTIAL_POLICIES, *GUARDED_POLICIES}:
        raise ValueError(f"unknown policy: {policy}")
    if kind not in {"qwen25vl", "uitars"}:
        raise ValueError(f"unknown model kind: {kind}")
    if attempt < 0:
        raise ValueError("attempt must be non-negative")
    if (
        not run_id
        or len(run_id) > 80
        or any(not (ch.isalnum() or ch in "-_.") for ch in run_id)
    ):
        raise ValueError("run_id must be a 1-80 character safe path component")
    if policy != "always" and (
        memory_goal is None or memory_params is None or memory is None
    ):
        raise ValueError("replay policies require successful memory")
    if policy in GUARDED_POLICIES and match_provider is None:
        raise ValueError("guarded replay policies require a match provider")
    if require_risk_certificate and policy != "full":
        raise ValueError("risk-limited activation applies only to full policy")
    if score_postcondition_for_ablations and policy not in GUARDED_POLICIES:
        raise ValueError(
            "scoring-only postconditions require a guarded replay policy"
        )

    spec = resolved_task_spec(task_id, timer_params, browser_task_seed)
    contract = CONTRACTS[contract_id]
    params_fingerprint = parameter_fingerprint(spec.params)
    episode_id = (
        f"{run_id}--{task_id}--p{params_fingerprint[:12]}--{contract_id}--"
        f"{kind}--{policy}--a{attempt:02d}"
    )
    episode_dir = output_root / episode_id
    C482.apply_contract(contract)
    env.reset(go_home=True)
    if task_id == "AudioRecorderRecordAudio":
        setup_apps.AudioRecorder.setup(env)
    task = task_registry[task_id](dict(spec.params))
    started = time.perf_counter()
    steps = []
    history_actions: list[dict[str, Any]] = []
    model_calls = 0
    reused_actions = 0
    confirmed_ids: set[str] = set()
    progress_recovered_ids: set[str] = set()
    invalidated_ids: set[str] = set()
    consumed_sequential_ids: list[str] = []
    sequential_cursor = 0
    replay_circuit_open = False
    failure_reason = None
    goal = None
    initial_reward = None
    final_reward = 0.0
    observation_ui_receipts: list[dict[str, Any]] = []
    try:
        task.initialize_task(env)
        observation_ui_receipts.append(
            ensure_clean_observation_ui(env, "after_task_initialization")
        )
        initial_reward = float(task.is_successful(env))
        open_app(env, spec.app_name)
        observation_ui_receipts.append(
            ensure_clean_observation_ui(env, "after_open_app")
        )
        state = env.get_state(wait_to_stabilize=True)
        goal = str(task.goal)
        if memory_goal is not None and goal != memory_goal:
            raise RuntimeError("Android exact-goal memory retrieval mismatch")
        if memory_params is not None and dict(spec.params) != dict(memory_params):
            raise RuntimeError("Android exact-parameter memory retrieval mismatch")
        current_raw = C482.pixels(state)
        current = C482.observe(current_raw, contract)

        for step in range(MAX_STEPS):
            before_path = episode_dir / f"step_{step:02d}_before.png"
            before_receipt = capture(before_path, current)
            history = history_text(history_actions)
            decision: dict[str, Any]
            context = None
            model_response = None
            action_xy = None
            action_source = None
            sequential_id = None
            guard_select_s = 0.0
            post_verify_s = 0.0
            scoring_only_post_verify_s = 0.0

            available_memory = [
                row
                for row in (memory or [])
                if row.transition_id not in invalidated_ids
            ]
            if policy in SEQUENTIAL_POLICIES:
                decision, action_xy = sequential_replay_decision(
                    policy, list(memory or []), sequential_cursor, current
                )
                if action_xy is not None:
                    action_source = "trajectory_replay"
                    sequential_id = str(decision["selected_transition_id"])
                    reused_actions += 1
            elif policy in GUARDED_POLICIES and not replay_circuit_open:
                guard_select_started = time.perf_counter()
                decision, context = GUARD.select(
                    available_memory,
                    current,
                    require_precondition=policy != "sg",
                    confirmed_transition_ids=confirmed_ids,
                    match_provider=match_provider,
                    risk_certificate=risk_certificate,
                    require_risk_certificate=require_risk_certificate,
                )
                guard_select_s = time.perf_counter() - guard_select_started
                recovered_now = set(
                    decision.get("progress_recovered_transition_ids", [])
                )
                known_memory_ids = {
                    row.transition_id for row in available_memory
                }
                if not recovered_now.issubset(known_memory_ids):
                    raise RuntimeError(
                        "guard recovered progress outside available memory"
                    )
                confirmed_ids.update(recovered_now)
                progress_recovered_ids.update(recovered_now)
            else:
                decision = {
                    "method_version": GUARD.METHOD_VERSION,
                    "decision": "fallback",
                    "selected_transition_id": None,
                    "selected_demonstration_order": None,
                    "mapped_action_xy": None,
                    "confirmed_transition_ids": sorted(confirmed_ids),
                    "evaluations": [],
                    "replay_circuit_open": replay_circuit_open,
                }

            if decision["decision"] == "reuse":
                if policy in GUARDED_POLICIES:
                    if context is None:
                        raise RuntimeError(
                            "reuse decision lacks authorization context"
                        )
                    action_xy = context.mapped_action_xy
                    action_source = "transition_guard"
                    reused_actions += 1
                elif policy not in SEQUENTIAL_POLICIES:
                    raise RuntimeError("unexpected reuse policy")
            else:
                request_id = f"{episode_id}--step-{step:02d}"
                response = GROUND_CLIENT.infer(
                    model_service_url,
                    request_id,
                    before_path,
                    instruction=instruction(goal, history),
                )
                model_response = compact_model_response(response)
                model_calls += 1
                parsed = response["result"]["parsed_action"]
                if not parsed.get("valid"):
                    failure_reason = "base_agent_action_parse_failure"
                else:
                    action_xy, clamped = normalized_to_pixel(
                        parsed["normalized_xy"], before_receipt
                    )
                    action_source = "base_agent"
                    decision["base_action_boundary_clamped"] = clamped

            if action_xy is None:
                steps.append(
                    {
                        "step": step,
                        "before": before_receipt,
                        "history": history,
                        "decision": decision,
                        "model_response": model_response,
                        "action_source": action_source,
                        "action_executed": False,
                        "action_xy": None,
                        "after": None,
                        "post_verification": None,
                        "scoring_only_post_verification": None,
                        "guard_select_s": guard_select_s,
                        "post_verify_s": post_verify_s,
                        "scoring_only_post_verify_s": (
                            scoring_only_post_verify_s
                        ),
                        "official_reward_after_action": None,
                    }
                )
                break

            requested_action_xy = action_xy
            action_xy = execute_click(env, requested_action_xy, current)
            decision["requested_action_xy"] = list(requested_action_xy)
            decision["executed_action_xy"] = list(action_xy)
            observation_ui_receipts.append(
                ensure_clean_observation_ui(env, f"after_step_{step:02d}_action")
            )
            state = env.get_state(wait_to_stabilize=True)
            next_raw = C482.pixels(state)
            next_observation = C482.observe(next_raw, contract)
            after_path = episode_dir / f"step_{step:02d}_after.png"
            after_receipt = capture(after_path, next_observation)
            post = None
            scoring_post = None
            if action_source == "trajectory_replay":
                if sequential_id is None:
                    raise RuntimeError("sequential replay lacks transition ID")
                consumed_sequential_ids.append(sequential_id)
                sequential_cursor += 1
            elif action_source == "transition_guard":
                if policy == "full":
                    post_started = time.perf_counter()
                    post = GUARD.verify_post(context, next_observation)
                    post_verify_s = time.perf_counter() - post_started
                    scoring_post = post
                elif score_postcondition_for_ablations:
                    scoring_started = time.perf_counter()
                    scoring_post = GUARD.verify_post(
                        context, next_observation
                    )
                    scoring_only_post_verify_s = (
                        time.perf_counter() - scoring_started
                    )
                if post is None or post["post_verified"]:
                    confirmed_ids.add(context.transition_id)
                else:
                    invalidated_ids.add(context.transition_id)
                    replay_circuit_open = True
            final_reward = float(task.is_successful(env))
            steps.append(
                {
                    "step": step,
                    "before": before_receipt,
                    "history": history,
                    "decision": decision,
                    "model_response": model_response,
                    "action_source": action_source,
                    "action_executed": True,
                    "action_xy": list(action_xy),
                    "after": after_receipt,
                    "post_verification": post,
                    "scoring_only_post_verification": scoring_post,
                    "guard_select_s": guard_select_s,
                    "post_verify_s": post_verify_s,
                    "scoring_only_post_verify_s": scoring_only_post_verify_s,
                    "official_reward_after_action": final_reward,
                }
            )
            history_actions.append(
                {"action_xy": list(action_xy), "source": action_source}
            )
            current = next_observation
            if final_reward == 1.0:
                break
        else:
            failure_reason = "maximum_steps_exhausted"

        success = final_reward == 1.0
        if not success and failure_reason is None:
            failure_reason = "official_task_incomplete"
        return {
            "runner_version": RUNNER_VERSION,
            "method_version": GUARD.METHOD_VERSION,
            "risk_certificate_required": require_risk_certificate,
            "risk_certificate": (
                risk_certificate.receipt()
                if risk_certificate is not None
                else None
            ),
            "paper_evidence": False,
            "episode_id": episode_id,
            "run_id": run_id,
            "attempt": attempt,
            "task_id": task_id,
            "params": spec.params,
            "parameter_fingerprint": params_fingerprint,
            "goal": goal,
            "contract": contract_id,
            "policy": policy,
            "kind": kind,
            "nonpoint_open_app_bootstrap": spec.app_name,
            "open_app_settle_s": OPEN_APP_SETTLE_S,
            "observation_ui": {
                "required_pointer_location": "0",
                "all_receipts_clean": all(
                    row["pointer_location_after_cleanup"] == "0"
                    for row in observation_ui_receipts
                ),
                "receipts": observation_ui_receipts,
            },
            "max_steps": MAX_STEPS,
            "initial_reward": initial_reward,
            "final_reward": final_reward,
            "success": success,
            "failure_reason": failure_reason,
            "executed_actions": sum(row["action_executed"] for row in steps),
            "model_calls": model_calls,
            "reused_actions": reused_actions,
            "postcondition_failures": sum(
                bool(
                    row["post_verification"]
                    and not row["post_verification"]["post_verified"]
                )
                for row in steps
            ),
            "scoring_only_postcondition_failures": sum(
                bool(
                    row.get("scoring_only_post_verification")
                    and not row["scoring_only_post_verification"][
                        "post_verified"
                    ]
                )
                for row in steps
            ),
            "operational_guard_s": sum(
                float(row.get("guard_select_s", 0.0))
                + float(row.get("post_verify_s", 0.0))
                for row in steps
            ),
            "scoring_only_audit_s": sum(
                float(row.get("scoring_only_post_verify_s", 0.0))
                for row in steps
            ),
            "score_postcondition_for_ablations": (
                score_postcondition_for_ablations
            ),
            "confirmed_transition_ids": sorted(confirmed_ids),
            "progress_recovered_transition_ids": sorted(
                progress_recovered_ids
            ),
            "consumed_sequential_transition_ids": consumed_sequential_ids,
            "invalidated_transition_ids": sorted(invalidated_ids),
            "replay_circuit_open": replay_circuit_open,
            "elapsed_s": time.perf_counter() - started,
            "steps": steps,
            "input_boundary": {
                "model": ["current screenshot", "goal", "pixel action history"],
                "guard": [
                    "successful source transition pixels/actions",
                    "current and next target screenshots",
                ],
                "excluded": [
                    "target accessibility tree",
                    "target element labels",
                    "target action oracle",
                    "official reward before post-action scoring",
                ],
            },
        }
    finally:
        task.tear_down(env)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=sorted(TASKS), required=True)
    parser.add_argument("--contract", choices=sorted(CONTRACTS), required=True)
    parser.add_argument(
        "--policy",
        choices=("always", "raw", "naive", "sg", "sgp", "full"),
        required=True,
    )
    parser.add_argument("--kind", choices=("qwen25vl", "uitars"), required=True)
    parser.add_argument("--model-service-url", required=True)
    parser.add_argument("--lightglue-url", default="http://127.0.0.1:48242")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument("--timer-hours", type=int)
    parser.add_argument("--timer-minutes", type=int)
    parser.add_argument("--timer-seconds", type=int)
    parser.add_argument("--browser-task-seed", type=int)
    parser.add_argument("--memory", type=Path)
    parser.add_argument("--risk-certificate", type=Path)
    parser.add_argument("--require-risk-certificate", action="store_true")
    parser.add_argument(
        "--score-postcondition-for-ablations", action="store_true"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT / "tmp" / "live_androidworld_transition_guard",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.policy != "always" and args.memory is None:
        parser.error("--memory is required for replay policies")
    if args.policy == "always" and args.memory is not None:
        parser.error("--memory is invalid for always policy")
    if args.require_risk_certificate and args.policy != "full":
        parser.error("--require-risk-certificate is valid only for full")
    if args.require_risk_certificate and args.risk_certificate is None:
        parser.error("required risk certificate path is missing")
    if args.risk_certificate is not None and args.policy != "full":
        parser.error("risk certificate is valid only for full policy")
    timer_values = (args.timer_hours, args.timer_minutes, args.timer_seconds)
    if any(value is not None for value in timer_values) and not all(
        value is not None for value in timer_values
    ):
        parser.error("all three --timer-* arguments must be supplied together")
    timer_params = (
        {
            "hours": args.timer_hours,
            "minutes": args.timer_minutes,
            "seconds": args.timer_seconds,
        }
        if all(value is not None for value in timer_values)
        else None
    )
    if args.task != "ClockTimerEntry" and timer_params is not None:
        parser.error("--timer-* arguments are valid only for ClockTimerEntry")
    if args.task != "BrowserMaze" and args.browser_task_seed is not None:
        parser.error("--browser-task-seed is valid only for BrowserMaze")

    model_health = GROUND_CLIENT.health(args.model_service_url)
    if not model_health.get("ready") or model_health.get("kind") != args.kind:
        raise RuntimeError("model service health/kind mismatch")
    memory_goal = None
    memory_params = None
    memory = None
    match_provider = None
    lightglue_health = None
    risk_certificate = None
    if args.policy != "always":
        memory_goal, memory_params, memory = load_memory(args.memory.resolve())
    if args.policy in GUARDED_POLICIES:
        lightglue = LightGlueServiceClient(args.lightglue_url)
        lightglue_health = lightglue.health()
        if not lightglue_health.get("ready"):
            raise RuntimeError("LightGlue service is not ready")
        match_provider = lightglue.correspondences
    if args.risk_certificate is not None:
        risk_certificate = load_risk_certificate(
            args.risk_certificate.resolve()
        )

    a11y_grpc_wrapper._get_accessibility_forwarder_apk = C459.A11Y_APK.read_bytes
    task_registry = registry.TaskRegistry().get_registry(
        registry.TaskRegistry.ANDROID_WORLD_FAMILY
    )
    env = None
    try:
        C482.restore_display()
        env = env_launcher.load_and_setup_env(
            console_port=5554,
            emulator_setup=False,
            freeze_datetime=True,
            adb_path=str(C459.ADB),
            grpc_port=8554,
        )
        env.hide_automation_ui()
        time.sleep(1.0)
        episode = run_episode(
            env=env,
            task_registry=task_registry,
            task_id=args.task,
            contract_id=args.contract,
            policy=args.policy,
            kind=args.kind,
            model_service_url=args.model_service_url,
            run_id=args.run_id,
            attempt=args.attempt,
            output_root=args.output_root.resolve(),
            timer_params=timer_params,
            browser_task_seed=args.browser_task_seed,
            memory_goal=memory_goal,
            memory_params=memory_params,
            memory=memory,
            match_provider=match_provider,
            risk_certificate=risk_certificate,
            require_risk_certificate=args.require_risk_certificate,
            score_postcondition_for_ablations=(
                args.score_postcondition_for_ablations
            ),
        )
    finally:
        C482.restore_display()
        if env is not None:
            env.close()
    episode["services"] = {
        "model": model_health,
        "lightglue": lightglue_health,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(episode, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(
        json.dumps(
            {
                key: episode[key]
                for key in (
                    "episode_id",
                    "success",
                    "failure_reason",
                    "executed_actions",
                    "model_calls",
                    "reused_actions",
                    "postcondition_failures",
                    "elapsed_s",
                )
            },
            indent=2,
        )
    )
    return 0 if episode["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
