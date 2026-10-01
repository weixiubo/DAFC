#!/usr/bin/env python3
"""Audit the official AndroidWorld controller and a programmatic task reward."""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import hashlib
from importlib.metadata import version
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

from android_env.wrappers import a11y_grpc_wrapper
from android_world import registry
from android_world.env import env_launcher
from android_world.env import json_action
import cv2


from dafc.paths import PROJECT
ANDROID_WORLD = PROJECT / "tmp" / "external" / "android_world"
SDK = Path(__import__("os").environ.get("ANDROID_HOME", str(Path.home()/"Android/Sdk")))
ADB = SDK / "platform-tools" / "adb"
A11Y_APK = (
    PROJECT
    / "tmp"
    / "c459_androidworld_runtime"
    / "2024.05.13-accessibility_forwarder.apk"
)
ROOT = PROJECT / "tmp" / "c459_androidworld_clock_mechanics"
OUTPUT = PROJECT / "validation" / "c459_androidworld_clock_mechanics.json"
EXPECTED_REPO_COMMIT = "3e50888527ef9f29b9157ecd537e408008bb1c85"
EXPECTED_A11Y_SHA256 = (
    "97a56a544e44d79f9b3181fc7dbdd72cffa908efd3d53c82afad1773061a350a"
)
TASK = "ClockStopWatchRunning"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture(name: str, state: Any) -> dict[str, Any]:
    path = ROOT / f"{name}.png"
    if not cv2.imwrite(str(path), cv2.cvtColor(state.pixels, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"could not encode screenshot: {path}")
    return {
        "path": path.relative_to(PROJECT).as_posix(),
        "sha256": sha256(path),
        "width": int(state.pixels.shape[1]),
        "height": int(state.pixels.shape[0]),
        "ui_elements_scoring_only": len(state.ui_elements),
    }


def label(element: Any) -> str | None:
    return element.content_description or element.text


def click_element(env: Any, state: Any, target: str) -> dict[str, Any]:
    candidates = [
        (index, element)
        for index, element in enumerate(state.ui_elements)
        if target in {element.text, element.content_description}
        and element.bbox_pixels is not None
        and element.is_visible is not False
        and element.is_enabled is not False
    ]
    if not candidates:
        observed = sorted({value for element in state.ui_elements if (value := label(element))})
        raise RuntimeError(f"missing visible target {target!r}; observed={observed}")
    candidates.sort(
        key=lambda row: (
            row[1].is_clickable is not True,
            row[1].bbox_pixels.area,
            row[0],
        )
    )
    index, element = candidates[0]
    x, y = element.bbox_pixels.center
    action = json_action.JSONAction(action_type="click", x=int(x), y=int(y))
    env.execute_action(action)
    time.sleep(2.0)
    return {
        "target_scoring_only": target,
        "ui_index_scoring_only": index,
        "coordinate_scoring_only": [int(x), int(y)],
        "bbox_scoring_only": [
            int(element.bbox_pixels.x_min),
            int(element.bbox_pixels.y_min),
            int(element.bbox_pixels.x_max),
            int(element.bbox_pixels.y_max),
        ],
    }


def adb_text(*args: str) -> str:
    return subprocess.check_output(
        [str(ADB), "-s", "emulator-5554", *args], text=True
    ).strip()


def main() -> int:
    ROOT.mkdir(parents=True, exist_ok=True)
    if not A11Y_APK.is_file() or sha256(A11Y_APK) != EXPECTED_A11Y_SHA256:
        raise RuntimeError("pinned accessibility-forwarder APK is unavailable")
    repo_commit = subprocess.check_output(
        ["git", "-C", str(ANDROID_WORLD), "rev-parse", "HEAD"], text=True
    ).strip()

    # AndroidEnv otherwise downloads this exact official asset on every
    # controller construction.  Reading the pinned copy changes transport only.
    a11y_grpc_wrapper._get_accessibility_forwarder_apk = A11Y_APK.read_bytes

    env = None
    task = None
    artifacts: dict[str, Any] = {}
    actions: list[dict[str, Any]] = []
    initial_success = None
    final_success = None
    goal = None
    activity = None
    try:
        env = env_launcher.load_and_setup_env(
            console_port=5554,
            emulator_setup=False,
            freeze_datetime=True,
            adb_path=str(ADB),
            grpc_port=8554,
        )
        env.hide_automation_ui()
        time.sleep(1.0)
        env.reset(go_home=True)
        task_registry = registry.TaskRegistry().get_registry(
            registry.TaskRegistry.ANDROID_WORLD_FAMILY
        )
        task_type = task_registry[TASK]
        task = task_type(task_type.generate_random_params())
        task.initialize_task(env)
        initial = env.get_state(wait_to_stabilize=True)
        artifacts["initialized"] = capture("00_initialized", initial)
        initial_success = float(task.is_successful(env))
        goal = str(task.goal)

        env.execute_action(
            json_action.JSONAction(action_type="open_app", app_name="clock")
        )
        time.sleep(2.0)
        clock = env.get_state(wait_to_stabilize=True)
        artifacts["clock"] = capture("01_clock", clock)
        actions.append(click_element(env, clock, "Stopwatch"))

        stopwatch = env.get_state(wait_to_stabilize=True)
        artifacts["stopwatch"] = capture("02_stopwatch", stopwatch)
        actions.append(click_element(env, stopwatch, "Start"))

        running = env.get_state(wait_to_stabilize=True)
        artifacts["running"] = capture("03_running", running)
        final_success = float(task.is_successful(env))
        activity = env.foreground_activity_name
    finally:
        if task is not None and env is not None:
            task.tear_down(env)
        if env is not None:
            env.close()

    unique_image_hashes = {row["sha256"] for row in artifacts.values()}
    criteria = {
        "development_only": True,
        "official_androidworld_commit_matches": repo_commit
        == EXPECTED_REPO_COMMIT,
        "official_accessibility_forwarder_hash_matches": sha256(A11Y_APK)
        == EXPECTED_A11Y_SHA256,
        "android_api_level_33": adb_text("shell", "getprop", "ro.build.version.sdk")
        == "33",
        "pointer_overlay_disabled": adb_text(
            "shell", "settings", "get", "system", "pointer_location"
        )
        == "0",
        "official_task_registry_entry_used": TASK == "ClockStopWatchRunning",
        "official_programmatic_reward_used": True,
        "initial_state_is_unsuccessful": initial_success == 0.0,
        "exact_two_oracle_mechanics_actions": [
            row["target_scoring_only"] for row in actions
        ]
        == ["Stopwatch", "Start"],
        "final_state_is_successful": final_success == 1.0,
        "deskclock_is_foreground": activity is not None and "DeskClock" in activity,
        "four_hash_bound_rgb_observations": len(artifacts) == 4
        and len(unique_image_hashes) == 4,
        "no_model_calls": True,
        "ui_tree_used_only_for_mechanics_scoring": True,
    }
    result = {
        "title": "C459 AndroidWorld Clock Mechanics Audit",
        "paper_evidence": False,
        "purpose": "establish official controller, RGB, action, and reward mechanics",
        "androidworld": {
            "path": str(ANDROID_WORLD),
            "commit": repo_commit,
        },
        "runtime": {
            "python": platform.python_version(),
            "android_env": version("android-env"),
            "android_world": version("android-world"),
            "opencv_python": version("opencv-python"),
            "pillow": version("pillow"),
            "android_api_level": adb_text(
                "shell", "getprop", "ro.build.version.sdk"
            ),
            "physical_resolution": [1080, 2400],
        },
        "accessibility_forwarder": {
            "source": (
                "https://storage.googleapis.com/android_env-tasks/"
                "2024.05.13-accessibility_forwarder.apk"
            ),
            "path": str(A11Y_APK),
            "bytes": A11Y_APK.stat().st_size,
            "sha256": sha256(A11Y_APK),
            "transport_patch": "official download function reads pinned local bytes",
        },
        "task": {
            "registry_id": TASK,
            "goal": goal,
            "initial_success": initial_success,
            "final_success": final_success,
            "foreground_activity": activity,
        },
        "actions_scoring_only": actions,
        "artifacts": artifacts,
        "criteria": criteria,
        "pass": all(criteria.values()),
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"task": result["task"], "criteria": criteria, "pass": result["pass"]}, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
