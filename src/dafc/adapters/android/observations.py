"""Closed-loop AndroidWorld runner for real GUI models and TransitionGuard.

The runner begins after a non-point ``open_app`` bootstrap shared by every
policy.  Base actions come from the persistent audited GUI-grounder service;
replay sees only successful source transition pixels/actions and current/next
target screenshots.  Android accessibility nodes are not used for action
selection, waiting, or history.  Official task reward is read only after an
executed action for episode scoring and successful-memory admission.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

import dafc.adapters.android.device as C482
import dafc.transitions as GUARD
from dafc.adapters.android.device import AndroidDevice
from dafc.paths import receipt_path

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
    LiveTaskSpec("ClockTimerEntry", {"hours": 0, "minutes": 1, "seconds": 30}, "clock"),
    LiveTaskSpec("SystemWifiTurnOn", {"on_or_off": "on"}, "settings"),
    LiveTaskSpec("SystemBluetoothTurnOn", {"on_or_off": "on"}, "settings"),
    LiveTaskSpec("SystemWifiTurnOff", {"on_or_off": "off"}, "settings"),
    LiveTaskSpec("SystemBluetoothTurnOff", {"on_or_off": "off"}, "settings"),
    LiveTaskSpec("BrowserMaze", {"browser_task_seed": 493001}, "files"),
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
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not encode {path}")
    return {
        "path": str(path.resolve()),
        "sha256": sha256(path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }


def history_text(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return "This is the task's initial in-app state."
    selected = actions[-HISTORY_HORIZON:]
    first_step = len(actions) - len(selected) + 1
    return "\n".join(
        (
            f"Step {first_step + offset}: tapped full-screenshot pixel coordinates ({row['action_xy'][0]:.1f}, {row['action_xy'][1]:.1f})."
            for offset, row in enumerate(selected)
        )
    )


def instruction(goal: str, history: str) -> str:
    return f"Original Android task: {goal}\nThe required Android app is already open.\nPreviously executed action history:\n{history}\nUsing only the current screenshot and this action history, select exactly one next executable tap that advances the original task. Do not repeat a completed action and do not explain the answer."


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
    if len(normalized_xy) != 2 or not all((math.isfinite(float(value)) for value in normalized_xy)):
        raise ValueError("model action must contain two finite coordinates")
    raw = (float(normalized_xy[0]) * float(image["width"]), float(normalized_xy[1]) * float(image["height"]))
    clamped = (
        min(max(raw[0], 0.0), float(image["width"]) - 1.0),
        min(max(raw[1], 0.0), float(image["height"]) - 1.0),
    )
    return (clamped, clamped != raw)


def execute_click(env: Any, point: tuple[float, float], observation: np.ndarray) -> tuple[float, float]:
    from android_world.env import json_action

    if not all((math.isfinite(value) for value in point)):
        raise ValueError("click coordinates must be finite")
    height, width = observation.shape[:2]
    executed = (min(max(int(round(point[0])), 0), width - 1), min(max(int(round(point[1])), 0), height - 1))
    env.execute_action(json_action.JSONAction(action_type="click", x=executed[0], y=executed[1]))
    time.sleep(2.0 + POST_ACTION_SETTLE_S)
    return (float(executed[0]), float(executed[1]))


def ensure_clean_observation_ui(env: Any, stage: str, *, device: AndroidDevice) -> dict[str, Any]:
    before = device.adb("shell", "settings", "get", "system", "pointer_location")
    env.hide_automation_ui()
    time.sleep(AUTOMATION_UI_SETTLE_S)
    after = device.adb("shell", "settings", "get", "system", "pointer_location")
    if after != "0":
        raise RuntimeError(f"Android automation UI remains enabled before observation: {after!r}")
    return {
        "stage": stage,
        "pointer_location_before_cleanup": before,
        "pointer_location_after_cleanup": after,
        "corrected": before != "0",
    }


def open_app(env: Any, app_name: str) -> None:
    from android_world.env import json_action

    env.execute_action(json_action.JSONAction(action_type="open_app", app_name=app_name))
    time.sleep(OPEN_APP_SETTLE_S)


def load_memory(
    path: Path, *, resource_base: Path | None = None
) -> tuple[str, dict[str, int | str], list[GUARD.Transition]]:
    episode = json.loads(path.read_text(encoding="utf-8"))
    executed = _admit_memory(episode)
    base = resource_base if resource_base is not None else path.parent
    transitions = []
    for index, row in enumerate(executed):
        before_path = receipt_path(row["before"]["path"], base=base)
        after_path = receipt_path(row["after"]["path"], base=base)
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
                action_xy=tuple((float(value) for value in row["action_xy"])),
                after=after,
                terminal=index == len(executed) - 1,
            )
        )
    params = episode.get("params")
    if not isinstance(params, dict):
        raise RuntimeError("Android source episode lacks task parameters")
    return (str(episode["goal"]), dict(params), transitions)


def load_risk_certificate(path: Path) -> GUARD.RiskCertificate:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "certificate" in payload and payload.get("pass") is not True:
        raise RuntimeError("risk-certificate qualification did not pass")
    row = payload.get("certificate", payload)
    certificate = GUARD.RiskCertificate.from_dict(row)
    if not certificate.active:
        raise RuntimeError("risk certificate is inactive")
    return certificate


def validated_timer_params(values: dict[str, Any] | None) -> dict[str, int] | None:
    if values is None:
        return None
    if set(values) != {"hours", "minutes", "seconds"}:
        raise ValueError("timer parameters must be exactly hours/minutes/seconds")
    if any((isinstance(value, bool) or not isinstance(value, int) for value in values.values())):
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
    task_id: str, timer_params: dict[str, Any] | None, browser_task_seed: Any | None = None
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
        return LiveTaskSpec(task_id=base.task_id, params=validated, app_name=base.app_name)
    if task_id == "BrowserMaze" and validated_seed is not None:
        return LiveTaskSpec(
            task_id=base.task_id, params={"browser_task_seed": validated_seed}, app_name=base.app_name
        )
    return base


def parameter_fingerprint(params: dict[str, Any]) -> str:
    encoded = json.dumps(params, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def sequential_replay_decision(
    policy: str, memory: list[GUARD.Transition], cursor: int, current: np.ndarray
) -> tuple[dict[str, Any], tuple[float, float] | None]:
    """Select the next stored action without inspecting current pixels."""
    if policy not in SEQUENTIAL_POLICIES:
        raise ValueError(f"not a sequential replay policy: {policy}")
    ordered = sorted(memory, key=lambda row: row.demonstration_order)
    if cursor >= len(ordered):
        return (
            {
                "method_version": "sequential-replay-baseline-v1",
                "baseline": policy,
                "decision": "fallback",
                "reason": "stored_trajectory_exhausted",
                "selected_transition_id": None,
                "selected_demonstration_order": None,
                "mapped_action_xy": None,
                "cursor": cursor,
                "pixel_content_inspected": False,
            },
            None,
        )
    transition = ordered[cursor]
    if not transition.source_success:
        return (
            {
                "method_version": "sequential-replay-baseline-v1",
                "baseline": policy,
                "decision": "fallback",
                "reason": "source_episode_not_successful",
                "selected_transition_id": transition.transition_id,
                "selected_demonstration_order": transition.demonstration_order,
                "mapped_action_xy": None,
                "cursor": cursor,
                "pixel_content_inspected": False,
            },
            None,
        )
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
    return (
        {
            "method_version": "sequential-replay-baseline-v1",
            "baseline": policy,
            "decision": "reuse",
            "reason": "absolute_source_pixel_sequence"
            if policy == "raw"
            else "dimension_normalized_source_sequence",
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
        },
        mapped,
    )


def _admit_memory(episode):
    if episode.get("runner_version") != RUNNER_VERSION:
        raise RuntimeError("Android source episode runner version mismatch")
    if episode.get("policy") != "always" or episode.get("contract") != "identity":
        raise RuntimeError("Android memory must be an identity always-policy source")
    if not episode.get("success"):
        raise RuntimeError("Android memory source episode was unsuccessful")
    executed = [row for row in episode["steps"] if row["action_executed"]]
    if not executed:
        raise RuntimeError("successful Android source has no point action")
    return executed
