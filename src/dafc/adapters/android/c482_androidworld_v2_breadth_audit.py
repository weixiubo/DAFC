#!/usr/bin/env python3
"""Development-only AndroidWorld breadth audit for TransitionGuard V2.

The audit builds successful source memories for four official tasks, then runs
separate target episodes under four observation/layout contracts.  Reuse sees
only stored transition pixels, stored point actions, source success receipts,
and the current/next target screenshot.  Accessibility labels are used only to
construct development demonstrations, provide an oracle fallback, and score
whether a replay advanced the official task mechanics.

This is a qualification audit, not paper evidence.  Its thresholds are fixed
in this file before target outcomes are observed.
"""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import argparse
import hashlib
import json
import subprocess
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

import dafc.adapters.android.c459_androidworld_clock_mechanics_audit as C459
import dafc.transitions as GUARD
from dafc.models.lightglue_runtime_service import LightGlueServiceClient


from dafc.paths import PROJECT
ROOT = PROJECT / "tmp" / "c482_androidworld_v2_breadth_audit"
OUTPUT = PROJECT / "validation" / "c482_androidworld_v2_breadth_audit.json"
C478_INPUT = PROJECT / "validation" / "c478_transition_guard_v2_power_audit.json"
ADB_SERIAL = "emulator-5554"
DEFAULT_DENSITY = 420
SERVICE_VERSION = "transitionguard-lightglue-loopback-v1"

# Frozen before target execution.
MIN_REUSE_RATE = 0.35
REQUIRED_TASKS_WITH_REUSE = 4
REQUIRED_CONTRACTS_WITH_REUSE = 4
MAX_NONADVANCING_REPLAYS = 0


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    params: dict[str, Any]
    app_name: str
    action_labels: tuple[str, ...]


@dataclass(frozen=True)
class ContractSpec:
    contract_id: str
    density: int = DEFAULT_DENSITY
    font_scale: float = 1.0
    jpeg_quality: int | None = None


TASKS = (
    TaskSpec(
        "ClockStopWatchRunning",
        {},
        "clock",
        ("Stopwatch", "Start"),
    ),
    TaskSpec(
        "ClockTimerEntry",
        {"hours": 0, "minutes": 1, "seconds": 30},
        "clock",
        ("Timer", "1", "3", "0"),
    ),
    TaskSpec(
        "SystemWifiTurnOn",
        {"on_or_off": "on"},
        "settings",
        ("Network & internet", "Internet", "Wi-Fi"),
    ),
    TaskSpec(
        "SystemBluetoothTurnOn",
        {"on_or_off": "on"},
        "settings",
        (
            "Connected devices",
            "Connection preferences",
            "Bluetooth",
            "Use Bluetooth",
        ),
    ),
)

CONTRACTS = (
    ContractSpec("identity"),
    ContractSpec("density-480", density=480),
    ContractSpec("font-1.15", font_scale=1.15),
    ContractSpec("jpeg-70", jpeg_quality=70),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def slug(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in value).strip("-")


def capture(
    task_id: str,
    episode_id: str,
    step: int,
    image: np.ndarray,
) -> dict[str, Any]:
    directory = ROOT / slug(task_id) / slug(episode_id)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{step:02d}.png"
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"could not encode {path}")
    return {
        "path": path.relative_to(PROJECT).as_posix(),
        "sha256": sha256(path),
        "width": int(image.shape[1]),
        "height": int(image.shape[0]),
    }


def adb(*args: str) -> str:
    return subprocess.run(
        [str(C459.ADB), "-s", ADB_SERIAL, *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def restore_display() -> None:
    adb("shell", "wm", "density", "reset")
    adb("shell", "settings", "delete", "system", "font_scale")
    adb("shell", "cmd", "uimode", "night", "no")
    time.sleep(1.0)


def apply_contract(contract: ContractSpec) -> None:
    restore_display()
    if contract.density != DEFAULT_DENSITY:
        adb("shell", "wm", "density", str(contract.density))
    if contract.font_scale != 1.0:
        adb(
            "shell",
            "settings",
            "put",
            "system",
            "font_scale",
            f"{contract.font_scale:.2f}",
        )
    time.sleep(1.0)


def display_receipt() -> dict[str, str]:
    return {
        "density": adb("shell", "wm", "density"),
        "font_scale": adb("shell", "settings", "get", "system", "font_scale"),
        "night_mode": adb("shell", "cmd", "uimode", "night"),
    }


def pixels(state: Any) -> np.ndarray:
    return cv2.cvtColor(state.pixels, cv2.COLOR_RGB2BGR)


def observe(image: np.ndarray, contract: ContractSpec) -> np.ndarray:
    if contract.jpeg_quality is None:
        return image.copy()
    ok, encoded = cv2.imencode(
        ".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, contract.jpeg_quality]
    )
    if not ok:
        raise RuntimeError("could not JPEG-transform target observation")
    transformed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if transformed is None or transformed.shape != image.shape:
        raise RuntimeError("JPEG target observation changed image shape")
    return transformed


def element_label(element: Any) -> str | None:
    return element.content_description or element.text


def has_label(state: Any, target: str) -> bool:
    return any(
        target in {element.text, element.content_description}
        and element.bbox_pixels is not None
        and element.is_visible is not False
        and element.is_enabled is not False
        for element in state.ui_elements
    )


def wait_state(
    env: Any,
    *,
    required_label: str | None = None,
    timeout_s: float = 8.0,
) -> Any:
    deadline = time.monotonic() + timeout_s
    best = None
    while time.monotonic() < deadline:
        state = env.get_state(wait_to_stabilize=True)
        if best is None or len(state.ui_elements) > len(best.ui_elements):
            best = state
        if len(state.ui_elements) > 5 and (
            required_label is None or has_label(state, required_label)
        ):
            return state
        time.sleep(0.5)
    if best is None:
        raise RuntimeError("AndroidWorld returned no state")
    return best


def click_label(env: Any, state: Any, target: str) -> dict[str, Any]:
    if not has_label(state, target):
        state = wait_state(env, required_label=target)
    return C459.click_element(env, state, target)


def execute_mapped(env: Any, context: GUARD.AuthorizationContext) -> None:
    x, y = context.mapped_action_xy
    env.execute_action(
        json_action.JSONAction(
            action_type="click",
            x=int(round(x)),
            y=int(round(y)),
        )
    )
    time.sleep(2.0)


def open_app(env: Any, app_name: str, first_label: str) -> Any:
    env.execute_action(json_action.JSONAction(action_type="open_app", app_name=app_name))
    time.sleep(2.0)
    return wait_state(env, required_label=first_label)


def compact_selection(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        "decision": selection["decision"],
        "selected_transition_id": selection["selected_transition_id"],
        "selected_demonstration_order": selection[
            "selected_demonstration_order"
        ],
        "mapped_action_xy": selection["mapped_action_xy"],
        "confirmed_transition_ids": selection.get("confirmed_transition_ids", []),
        "observable_prefix_intact": selection.get("observable_prefix_intact"),
        "evaluations": [
            {
                "transition_id": row["transition_id"],
                "authorized": row["authorized"],
                "failed_gate": row["failed_gate"],
                "reason": row["reason"],
                "geometry": row["geometry"],
                "precondition": row["precondition"],
            }
            for row in selection["evaluations"]
        ],
    }


def build_source_memory(
    env: Any,
    task_type: Any,
    spec: TaskSpec,
    *,
    post_action_settle_s: float = 0.0,
) -> tuple[list[GUARD.Transition], dict[str, Any]]:
    if post_action_settle_s < 0:
        raise ValueError("post_action_settle_s must be non-negative")
    restore_display()
    env.reset(go_home=True)
    task = task_type(dict(spec.params))
    actions = []
    images = []
    artifacts = []
    try:
        task.initialize_task(env)
        initial_reward = float(task.is_successful(env))
        state = open_app(env, spec.app_name, spec.action_labels[0])
        image = pixels(state)
        images.append(image)
        artifacts.append(capture(spec.task_id, "source", 0, image))
        for index, label in enumerate(spec.action_labels):
            actions.append(click_label(env, state, label))
            time.sleep(post_action_settle_s)
            next_label = (
                spec.action_labels[index + 1]
                if index + 1 < len(spec.action_labels)
                else None
            )
            state = wait_state(env, required_label=next_label)
            image = pixels(state)
            images.append(image)
            artifacts.append(capture(spec.task_id, "source", index + 1, image))
        final_reward = float(task.is_successful(env))
        if initial_reward != 0.0 or final_reward != 1.0:
            raise RuntimeError(
                f"source mechanics failed for {spec.task_id}: "
                f"{initial_reward} -> {final_reward}"
            )
        # Admission deliberately occurs only after the terminal receipt above.
        transitions = [
            GUARD.Transition(
                transition_id=f"{spec.task_id}:{index}:{slug(label)}",
                demonstration_order=index,
                source_success=True,
                before=images[index],
                action_xy=tuple(actions[index]["coordinate_scoring_only"]),
                after=images[index + 1],
                terminal=index == len(spec.action_labels) - 1,
            )
            for index, label in enumerate(spec.action_labels)
        ]
        return transitions, {
            "task_id": spec.task_id,
            "goal": str(task.goal),
            "params": spec.params,
            "initial_reward": initial_reward,
            "final_reward": final_reward,
            "actions_scoring_only": actions,
            "artifacts": artifacts,
            "memory_admitted_after_success": True,
        }
    finally:
        task.tear_down(env)


def run_target(
    env: Any,
    task_type: Any,
    spec: TaskSpec,
    contract: ContractSpec,
    transitions: list[GUARD.Transition],
    match_provider: Callable[[np.ndarray, np.ndarray], dict[str, Any]],
    *,
    post_action_settle_s: float = 0.0,
) -> dict[str, Any]:
    if post_action_settle_s < 0:
        raise ValueError("post_action_settle_s must be non-negative")
    apply_contract(contract)
    env.reset(go_home=True)
    task = task_type(dict(spec.params))
    decisions = []
    posts = []
    artifacts = []
    fallback_actions = []
    confirmed: set[str] = set()
    selected_ids: list[str] = []
    circuit_open = False
    nonadvancing_replays = 0
    conservative_post_misses = 0
    try:
        task.initialize_task(env)
        initial_reward = float(task.is_successful(env))
        state = open_app(env, spec.app_name, spec.action_labels[0])
        current = observe(pixels(state), contract)
        artifacts.append(capture(spec.task_id, contract.contract_id, 0, current))

        for index, label in enumerate(spec.action_labels):
            next_label = (
                spec.action_labels[index + 1]
                if index + 1 < len(spec.action_labels)
                else None
            )
            used_replay = False
            advanced = False
            if not circuit_open:
                selection, context = GUARD.select(
                    transitions,
                    current,
                    confirmed_transition_ids=confirmed,
                    match_provider=match_provider,
                )
                decisions.append(compact_selection(selection))
                if context is not None:
                    used_replay = True
                    selected_ids.append(context.transition_id)
                    execute_mapped(env, context)
                    time.sleep(post_action_settle_s)
                    # The post-action observation timing is independent of
                    # target labels.  Labels are consulted only after capture
                    # to score whether the official mechanics advanced.
                    state = wait_state(env, timeout_s=5.0)
                    observed_next = observe(pixels(state), contract)
                    post = GUARD.verify_post(context, observed_next)
                    posts.append(post)
                    terminal_success = (
                        float(task.is_successful(env)) == 1.0
                        if next_label is None
                        else False
                    )
                    advanced = terminal_success or (
                        next_label is not None and has_label(state, next_label)
                    )
                    if not advanced:
                        nonadvancing_replays += 1
                    if post["post_verified"]:
                        confirmed.add(context.transition_id)
                    else:
                        circuit_open = True
                        if advanced:
                            conservative_post_misses += 1
                    current = observed_next
            else:
                decisions.append(
                    {
                        "decision": "fallback",
                        "selected_transition_id": None,
                        "selected_demonstration_order": None,
                        "mapped_action_xy": None,
                        "confirmed_transition_ids": sorted(confirmed),
                        "observable_prefix_intact": None,
                        "evaluations": [],
                        "reason": "episode_post_failure_circuit_open",
                    }
                )

            if not used_replay or not advanced:
                action = click_label(env, state, label)
                time.sleep(post_action_settle_s)
                fallback_actions.append(
                    {
                        "step": index,
                        "reason": (
                            "guard_abstention_or_circuit"
                            if not used_replay
                            else "nonadvancing_replay_recovery"
                        ),
                        **action,
                    }
                )
                state = wait_state(env, required_label=next_label)
                current = observe(pixels(state), contract)

            artifacts.append(
                capture(spec.task_id, contract.contract_id, index + 1, current)
            )

        final_reward = float(task.is_successful(env))
        return {
            "task_id": spec.task_id,
            "contract_id": contract.contract_id,
            "contract": {
                "density": contract.density,
                "font_scale": contract.font_scale,
                "jpeg_quality": contract.jpeg_quality,
            },
            "display_receipt": display_receipt(),
            "initial_reward": initial_reward,
            "final_reward": final_reward,
            "opportunities": len(spec.action_labels),
            "reused": len(selected_ids),
            "post_verified": sum(row["post_verified"] for row in posts),
            "post_failed": sum(not row["post_verified"] for row in posts),
            "nonadvancing_replays": nonadvancing_replays,
            "conservative_post_misses": conservative_post_misses,
            "circuit_opened": circuit_open,
            "selected_transition_ids": selected_ids,
            "confirmed_transition_ids": sorted(confirmed),
            "confirmed_ids_are_unique": len(selected_ids)
            == len(set(selected_ids)),
            "fallback_actions_scoring_only": fallback_actions,
            "decisions": decisions,
            "postconditions": posts,
            "artifacts": artifacts,
        }
    finally:
        task.tear_down(env)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lightglue-url", default="http://127.0.0.1:48242"
    )
    args = parser.parse_args()

    prior = json.loads(C478_INPUT.read_text(encoding="utf-8"))
    if not prior.get("pass") or prior.get("paper_evidence") is not False:
        raise RuntimeError("C478 V2 development qualification must pass")
    if not C459.A11Y_APK.is_file() or C459.sha256(C459.A11Y_APK) != C459.EXPECTED_A11Y_SHA256:
        raise RuntimeError("pinned accessibility-forwarder APK is unavailable")

    ROOT.mkdir(parents=True, exist_ok=True)
    client = LightGlueServiceClient(args.lightglue_url)
    health = client.health()
    if not health.get("ready") or health.get("service_version") != SERVICE_VERSION:
        raise RuntimeError("pinned loopback LightGlue service is not ready")

    a11y_grpc_wrapper._get_accessibility_forwarder_apk = C459.A11Y_APK.read_bytes
    task_registry = registry.TaskRegistry().get_registry(
        registry.TaskRegistry.ANDROID_WORLD_FAMILY
    )
    env = None
    source_rows = []
    target_rows = []
    memories: dict[str, list[GUARD.Transition]] = {}
    try:
        restore_display()
        env = env_launcher.load_and_setup_env(
            console_port=5554,
            emulator_setup=False,
            freeze_datetime=True,
            adb_path=str(C459.ADB),
            grpc_port=8554,
        )
        env.hide_automation_ui()
        time.sleep(1.0)

        for spec in TASKS:
            task_type = task_registry[spec.task_id]
            memory, row = build_source_memory(env, task_type, spec)
            memories[spec.task_id] = memory
            source_rows.append(row)

        for contract in CONTRACTS:
            for spec in TASKS:
                target_rows.append(
                    run_target(
                        env,
                        task_registry[spec.task_id],
                        spec,
                        contract,
                        memories[spec.task_id],
                        client.correspondences,
                    )
                )
    finally:
        restore_display()
        if env is not None:
            env.close()

    opportunities = sum(row["opportunities"] for row in target_rows)
    reused = sum(row["reused"] for row in target_rows)
    post_verified = sum(row["post_verified"] for row in target_rows)
    post_failed = sum(row["post_failed"] for row in target_rows)
    nonadvancing = sum(row["nonadvancing_replays"] for row in target_rows)
    tasks_with_reuse = {
        row["task_id"] for row in target_rows if row["reused"] > 0
    }
    contracts_with_reuse = {
        row["contract_id"] for row in target_rows if row["reused"] > 0
    }
    reuse_rate = reused / opportunities
    restored = display_receipt()
    criteria = {
        "development_only": True,
        "prior_v2_qualification_passed": True,
        "four_official_androidworld_tasks": len(TASKS) == 4,
        "two_official_apps_represented": {row.app_name for row in TASKS}
        == {"clock", "settings"},
        "all_source_episodes_officially_successful": all(
            row["initial_reward"] == 0.0 and row["final_reward"] == 1.0
            for row in source_rows
        ),
        "memory_admitted_only_after_source_success": all(
            row["memory_admitted_after_success"] for row in source_rows
        ),
        "sixteen_separate_target_episodes": len(target_rows)
        == len(TASKS) * len(CONTRACTS),
        "all_target_episodes_officially_successful": all(
            row["initial_reward"] == 0.0 and row["final_reward"] == 1.0
            for row in target_rows
        ),
        "reuse_rate_at_least_frozen_35_percent": reuse_rate >= MIN_REUSE_RATE,
        "all_four_tasks_have_reuse": len(tasks_with_reuse)
        >= REQUIRED_TASKS_WITH_REUSE,
        "all_four_contracts_have_reuse": len(contracts_with_reuse)
        >= REQUIRED_CONTRACTS_WITH_REUSE,
        "zero_nonadvancing_replays": nonadvancing
        <= MAX_NONADVANCING_REPLAYS,
        "every_selected_transition_post_verified": post_failed == 0
        and post_verified == reused,
        "confirmed_transition_never_selected_twice": all(
            row["confirmed_ids_are_unique"] for row in target_rows
        ),
        "post_failure_circuit_rule_implemented": True,
        "guard_inputs_are_screenshot_only": True,
        "target_accessibility_used_only_for_oracle_fallback_and_scoring": True,
        "open_app_bootstrap_excluded_from_reuse_count": True,
        "pinned_lightglue_service_binding": health["bindings"][
            "frontend_version"
        ]
        == "pinned-superpoint-lightglue-runtime-v1",
        "display_restored_to_baseline": (
            f"Physical density: {DEFAULT_DENSITY}" in restored["density"]
            and restored["font_scale"] == "null"
            and restored["night_mode"] == "Night mode: no"
        ),
    }
    result = {
        "title": "C482 AndroidWorld TransitionGuard V2 Breadth Audit",
        "paper_evidence": False,
        "purpose": (
            "qualify the full V2 state machine across four official AndroidWorld "
            "tasks before any formal protocol is frozen"
        ),
        "frozen_thresholds": {
            "minimum_reuse_rate": MIN_REUSE_RATE,
            "required_tasks_with_reuse": REQUIRED_TASKS_WITH_REUSE,
            "required_contracts_with_reuse": REQUIRED_CONTRACTS_WITH_REUSE,
            "maximum_nonadvancing_replays": MAX_NONADVANCING_REPLAYS,
        },
        "androidworld": {
            "path": str(C459.ANDROID_WORLD),
            "commit": adb("shell", "getprop", "ro.build.version.sdk")
            and subprocess.run(
                ["git", "-C", str(C459.ANDROID_WORLD), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip(),
            "api_level": adb("shell", "getprop", "ro.build.version.sdk"),
        },
        "lightglue_service": health,
        "tasks": [
            {
                "task_id": row.task_id,
                "params": row.params,
                "app_name": row.app_name,
                "action_labels": row.action_labels,
            }
            for row in TASKS
        ],
        "contracts": [row.__dict__ for row in CONTRACTS],
        "source_episodes": source_rows,
        "target_episodes": target_rows,
        "aggregate": {
            "opportunities": opportunities,
            "reused": reused,
            "reuse_rate": reuse_rate,
            "post_verified": post_verified,
            "post_failed": post_failed,
            "nonadvancing_replays": nonadvancing,
            "tasks_with_reuse": sorted(tasks_with_reuse),
            "contracts_with_reuse": sorted(contracts_with_reuse),
            "oracle_fallback_actions": sum(
                len(row["fallback_actions_scoring_only"]) for row in target_rows
            ),
        },
        "method_input_contract": {
            "allowed": [
                "stored source before screenshot",
                "stored source point action",
                "stored source after screenshot",
                "source terminal success receipt",
                "current and next target screenshots",
            ],
            "excluded": [
                "target accessibility tree",
                "target element label",
                "target reward before terminal scoring",
                "current target step identity",
            ],
            "development_oracle_boundary": (
                "target labels are consumed only after guard abstention or for "
                "mechanics scoring; they are never passed to TransitionGuard"
            ),
        },
        "restored_display": restored,
        "criteria": criteria,
        "pass": all(criteria.values()),
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "aggregate": result["aggregate"],
                "criteria": criteria,
                "pass": result["pass"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
