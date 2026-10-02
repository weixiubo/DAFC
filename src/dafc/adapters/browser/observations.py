"""Run one real MiniWoB episode with an online GUI model or TransitionGuard.

Every fallback action is generated from the screenshot actually observed in
the live episode through the persistent V5 grounder service.  The runner never
precomputes actions from an oracle trajectory.  DOM-derived task plans and
boxes are absent; MiniWoB reward is read only after actions for episode scoring
and source-success admission.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import numpy as np

import dafc.adapters.browser.capture as C388
import dafc.adapters.browser.contracts as C432
import dafc.models.client as CLIENT
import dafc.transitions as GUARD
from dafc.paths import receipt_path

RUNNER_VERSION = "live-miniwob-transition-guard-v9-external-artifacts"
HISTORY_HORIZON = 4
MAX_STEPS = 10
TASKS = (
    "choose-date-easy",
    "click-button-sequence",
    "click-collapsible",
    "click-option",
    "click-pie-nodelay",
    "click-dialog-2",
    "click-menu-2",
    "click-collapsible-2",
)
SEQUENTIAL_POLICIES = frozenset({"raw", "naive"})
GUARDED_POLICIES = frozenset({"sg", "sgp", "full"})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_contract() -> dict[str, Any]:
    return {
        **C388.source_contract(),
        "script": "",
        "supported": True,
        "exact_global": True,
        "capture_revision": "source-160x210-scale3-v1",
    }


def contract_by_name(name: str) -> dict[str, Any]:
    if name == "source":
        return source_contract()
    matches = [row for row in C432.multistep_contracts(124) if row["name"] == name]
    if len(matches) != 1:
        raise KeyError(f"unknown contract: {name}")
    return matches[0]


def history_text(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return "This is the task's initial state."
    selected = actions[-HISTORY_HORIZON:]
    first_step = len(actions) - len(selected) + 1
    return "\n".join(
        (
            f"Step{first_step + offset}: clicked at full-screenshot pixel coordinates ({row['action_image_xy'][0]:.1f},{row['action_image_xy'][1]:.1f})."
            for offset, row in enumerate(selected)
        )
    )


def generic_instruction(goal: str, history: str) -> str:
    return f"Original task: {goal}\nPreviously executed action history:\n{history}\nUsing the current screenshot and action history, select exactly one next executable click that advances the original task. Do not repeat a completed action."


def normalized_to_image(
    normalized_xy: list[float], image: dict[str, Any]
) -> tuple[tuple[float, float], bool]:
    width = float(image["width"])
    height = float(image["height"])
    raw = (float(normalized_xy[0]) * width, float(normalized_xy[1]) * height)
    clamped = (min(max(raw[0], 0.0), max(width - 1.0, 0.0)), min(max(raw[1], 0.0), max(height - 1.0, 0.0)))
    return (clamped, clamped != raw)


def sequential_replay_decision(
    policy: str, memory: list[Any], cursor: int, current: np.ndarray
) -> tuple[dict[str, Any], tuple[float, float] | None]:
    """Select the next trajectory action without inspecting current pixels."""
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


def image_to_css(action_image: tuple[float, float], contract: dict[str, Any]) -> list[float]:
    scale = float(contract["capture_scale"])
    return [action_image[0] / scale, action_image[1] / scale]


def load_memory(path: Path, *, resource_base: Path | None = None) -> tuple[str, list[Any]]:
    episode = json.loads(path.read_text(encoding="utf-8"))
    if episode.get("runner_version") != RUNNER_VERSION:
        raise RuntimeError("source episode runner version mismatch")
    if episode.get("contract") != "source":
        raise RuntimeError("memory source must use the source contract")
    if episode.get("policy") != "always":
        raise RuntimeError("memory source must use the always-call policy")
    if not episode.get("success"):
        raise RuntimeError("memory source episode did not succeed")
    base = resource_base if resource_base is not None else path.parent
    transitions = []
    source_rows = episode["steps"]
    for row in source_rows:
        if not row.get("action_executed"):
            raise RuntimeError("successful source contains a step without action")
        before_path = receipt_path(row["before"]["path"], base=base)
        after_path = receipt_path(row["after"]["path"], base=base)
        if sha256(before_path) != row["before"]["sha256"]:
            raise RuntimeError("source before-image hash mismatch")
        if sha256(after_path) != row["after"]["sha256"]:
            raise RuntimeError("source after-image hash mismatch")
        before = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
        after = cv2.imread(str(after_path), cv2.IMREAD_COLOR)
        if before is None or after is None:
            raise RuntimeError("could not load source transition images")
        transitions.append(
            GUARD.Transition(
                transition_id=f"{episode['episode_id']}--transition-{row['step']:02d}",
                demonstration_order=int(row["step"]),
                source_success=True,
                before=before,
                action_xy=tuple((float(value) for value in row["action_image_xy"])),
                after=after,
                terminal=int(row["step"]) == len(source_rows) - 1,
            )
        )
    return (str(episode["goal"]), transitions)


def load_risk_certificate(path: Path) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if "certificate" in payload and payload.get("pass") is not True:
        raise RuntimeError("risk-certificate qualification did not pass")
    row = payload.get("certificate", payload)
    certificate = GUARD.RiskCertificate.from_dict(row)
    if not certificate.active:
        raise RuntimeError("risk certificate is inactive")
    return certificate


def service_grounder(
    service_url: str, kind: str, request_id: str, image_path: Path, goal: str, history: str
) -> dict[str, Any]:
    if kind == "har":
        return CLIENT.infer(service_url, request_id, image_path, goal=goal, history=history)
    return CLIENT.infer(service_url, request_id, image_path, instruction=generic_instruction(goal, history))


def compact_service_response(response: dict[str, Any]) -> dict[str, Any]:
    result = response["result"]
    return {
        "service_version": response.get("service_version"),
        "kind": response.get("kind"),
        "request_sha256": response.get("request_sha256"),
        "service_s": response.get("service_s"),
        "result": {
            "id": result.get("id"),
            "image_sha256": result.get("image_sha256"),
            "image_width": result.get("image_width"),
            "image_height": result.get("image_height"),
            "instruction": result.get("instruction"),
            "goal": result.get("goal"),
            "history": result.get("history"),
            "prompt": result.get("prompt"),
            "raw_response": result.get("raw_response"),
            "parsed_action": result.get("parsed_action"),
            "inference_s": result.get("inference_s"),
            "input_tokens": result.get("input_tokens"),
            "generated_tokens": result.get("generated_tokens"),
        },
    }


def outcome_receipt(outcome: dict[str, Any]) -> dict[str, Any]:
    return {
        "reward": float(outcome["reward"]),
        "raw_reward": outcome.get("raw_reward"),
        "terminated": bool(outcome["terminated"]),
        "truncated": bool(outcome["truncated"]),
        "done": bool(outcome["done"]),
    }


if TYPE_CHECKING:
    pass
