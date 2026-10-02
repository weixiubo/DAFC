#!/usr/bin/env python3
"""Prospective native-Android atomic typed-reuse experiment.

The manifest stage is target-outcome blind.  Runtime jobs use accessibility
only to construct a fixed final-decision state; policy authorization and the
target actor receive pixels and the task instruction only.
"""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import argparse
import fcntl
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


from dafc.paths import PROJECT
SCRIPTS = PROJECT / "scripts"
SPECIFICATION = PROJECT / "reports/c585_android_atomic_closed_loop_specification.md"
SOURCE_REGISTRY = PROJECT / 'data/formal_transitionguard_android_recovered_v2/androidworld/recovered_source/registry.json'
CHECKPOINT_ATTESTATIONS = PROJECT / "formal/remote_checkpoint_attestations.json"
DEFAULT_MANIFEST = PROJECT / "formal/transitionguard_android_atomic_v3_manifest.json"
DEFAULT_SIGNATURE_JSON = PROJECT / "validation/c586c_android_atomic_manifest_v3_signature.json"
DEFAULT_SIGNATURE_REPORT = PROJECT / "reports/c586c_android_atomic_manifest_v3_signature.md"
DEFAULT_ROOT = PROJECT / 'data/formal_android_atomic_c587_v3'
NAMESPACE = "tg-aaai27-c587-android-atomic-v3"
MANIFEST_VERSION = "android-atomic-closed-loop-manifest-v3-refactored-v1"
RUNNER_VERSION = "android-atomic-closed-loop-runner-v3-refactored-v1"
PINNED_A11Y_APK = PROJECT / 'resources/external/2024.05.13-accessibility_forwarder.apk'
PINNED_A11Y_SHA256 = (
    "97a56a544e44d79f9b3181fc7dbdd72cffa908efd3d53c82afad1773061a350a"
)
from dafc.routing import (
    POLICIES as POLICIES,
    STANDARD_RNG_SEED as STANDARD_RNG_SEED,
    STANDARD_RANSAC_THRESHOLD_PX as STANDARD_RANSAC_THRESHOLD_PX,
    STANDARD_MAX_ITERS as STANDARD_MAX_ITERS,
    STANDARD_CONFIDENCE as STANDARD_CONFIDENCE,
    STANDARD_MIN_INLIERS as STANDARD_MIN_INLIERS,
    STANDARD_MIN_INLIER_RATIO as STANDARD_MIN_INLIER_RATIO,
    TARGET_BOUNDARY_MARGIN_PX as TARGET_BOUNDARY_MARGIN_PX,
    apply_homography as apply_homography,
    point_in_bounds as point_in_bounds,
    standard_registration as standard_registration,
    compact_dafc as compact_dafc,
    authorize as authorize,
)
ACTORS = ("qwen25vl", "uitars")
CONTRACTS = ("identity", "jpeg-70", "density-480", "font-1.15")
REPETITIONS = 5
POST_ACTION_SETTLE_S = 4.0


@dataclass(frozen=True)
class AtomicTask:
    task_id: str
    params: dict[str, int | str]
    app_name: str
    setup_prefix_labels: tuple[str, ...]
    terminal_label_scoring_only: str


TASKS = (
    AtomicTask("ClockStopWatchRunning", {}, "clock", ("Stopwatch",), "Start"),
    AtomicTask(
        "ClockTimerEntry",
        {"hours": 0, "minutes": 1, "seconds": 30},
        "clock",
        ("Timer", "1", "3"),
        "0",
    ),
    AtomicTask(
        "SystemWifiTurnOn",
        {"on_or_off": "on"},
        "settings",
        ("Network & internet", "Internet"),
        "Wi-Fi",
    ),
    AtomicTask(
        "SystemBluetoothTurnOn",
        {"on_or_off": "on"},
        "settings",
        ("Connected devices", "Connection preferences", "Bluetooth"),
        "Use Bluetooth",
    ),
    AtomicTask(
        "SystemWifiTurnOff",
        {"on_or_off": "off"},
        "settings",
        ("Network & internet", "Internet"),
        "Wi-Fi",
    ),
    AtomicTask(
        "SystemBluetoothTurnOff",
        {"on_or_off": "off"},
        "settings",
        ("Connected devices", "Connection preferences", "Bluetooth"),
        "Use Bluetooth",
    ),
    AtomicTask(
        "AudioRecorderRecordAudio",
        {},
        "audio recorder",
        (),
        "Recording: %s",
    ),
)
TASK_BY_ID = {row.task_id: row for row in TASKS}


if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import dafc.registration.selector_v5 as V5
import dafc.registration.action_contract as V6
C459 = LazyModule('dafc.adapters.android.c459_androidworld_clock_mechanics_audit')
C482 = LazyModule('dafc.adapters.android.c482_androidworld_v2_breadth_audit')
import dafc.models.client as GROUND_CLIENT


from dafc.storage import (
    sha256 as sha256,
    canonical_sha256 as canonical_sha256,
    load_json as load_json,
    write_new_json as write_new_json,
    write_new_text as write_new_text,
    image_receipt as image_receipt,
)




def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(PROJECT.resolve()).as_posix()




def git_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=PROJECT,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0 and not result.stdout


def committed(path: Path) -> bool:
    path = Path(path).resolve()
    try:
        name = path.relative_to(PROJECT.resolve()).as_posix()
    except ValueError:
        return False
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", name],
        cwd=PROJECT,
        capture_output=True,
        check=False,
    )
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", name],
        cwd=PROJECT,
        check=False,
    )
    return tracked.returncode == 0 and clean.returncode == 0


def git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT, text=True
    ).strip()


def git_tree(revision: str = "HEAD") -> str:
    return subprocess.check_output(
        ["git", "rev-parse", f"{revision}^{{tree}}"], cwd=PROJECT, text=True
    ).strip()


def resolve_receipt_path(value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (PROJECT / path).resolve()


def source_rows() -> dict[str, dict[str, Any]]:
    registry = load_json(SOURCE_REGISTRY)
    successes = {
        row["task"]: {**row, "source_success": True}
        for row in registry["selected_successes"]
    }
    failures = {
        row["task"]: {**row, "source_success": False}
        for row in registry["retained_failures"]
    }
    rows = {**successes, **failures}
    if set(rows) != set(TASK_BY_ID) or len(rows) != 7:
        raise RuntimeError("sealed seven-task source population changed")
    if set(successes) & set(failures):
        raise RuntimeError("source task appears in success and failure sets")
    return rows


def validate_source(row: dict[str, Any]) -> dict[str, Any]:
    artifact = Path(row["artifact"]).resolve()
    if not artifact.is_file() or sha256(artifact) != row["artifact_sha256"]:
        raise RuntimeError(f"source artifact hash mismatch: {artifact}")
    episode = load_json(artifact)
    if (
        episode.get("task_id") != row["task"]
        or episode.get("goal") != row["goal"]
        or episode.get("params") != row["params"]
        or bool(episode.get("success")) != bool(row["source_success"])
        or float(episode.get("final_reward", -1)) != float(row["official_reward"])
        or episode.get("policy") != "always"
        or episode.get("contract") != "identity"
    ):
        raise RuntimeError(f"source registry/episode mismatch: {row['task']}")
    base = {
        "task": row["task"],
        "goal": row["goal"],
        "params": row["params"],
        "source_success": bool(row["source_success"]),
        "artifact": str(artifact),
        "artifact_sha256": row["artifact_sha256"],
        "episode_id": row["episode_id"],
        "source_model_calls": int(row["model_calls"]),
        "source_official_reward": float(row["official_reward"]),
    }
    if not row["source_success"]:
        if episode.get("success") or float(episode.get("final_reward", -1)) != 0.0:
            raise RuntimeError("failed source unexpectedly succeeded")
        return {
            **base,
            "atomic_memory": None,
            "source_failure_reason": episode.get("failure_reason"),
        }

    executed = [step for step in episode["steps"] if step.get("action_executed")]
    rewards = [float(step["official_reward_after_action"]) for step in executed]
    if (
        not executed
        or rewards[-1] != 1.0
        or any(value != 0.0 for value in rewards[:-1])
        or float(episode.get("initial_reward", -1)) != 0.0
    ):
        raise RuntimeError("successful source lacks a unique terminal action receipt")
    terminal = executed[-1]
    before = resolve_receipt_path(terminal["before"]["path"])
    after = resolve_receipt_path(terminal["after"]["path"])
    for path, receipt in ((before, terminal["before"]), (after, terminal["after"])):
        if not path.is_file() or sha256(path) != receipt["sha256"]:
            raise RuntimeError(f"source transition image mismatch: {path}")
    action = [float(value) for value in terminal["action_xy"]]
    if len(action) != 2 or not all(math.isfinite(value) for value in action):
        raise RuntimeError("source terminal action is not a finite point")
    return {
        **base,
        "atomic_memory": {
            "source_before": str(before),
            "source_before_sha256": terminal["before"]["sha256"],
            "source_after": str(after),
            "source_after_sha256": terminal["after"]["sha256"],
            "source_action_xy": action,
            "source_step": int(terminal["step"]),
            "official_reward_after_action": 1.0,
            "earlier_executed_actions": len(executed) - 1,
            "earlier_rewards_all_zero": True,
        },
        "source_failure_reason": None,
    }


def job_key(actor: str, task: str, contract: str, policy: str, repetition: int) -> str:
    return f"{actor}|{task}|{contract}|{policy}|r{repetition:02d}"


def job_id(actor: str, task: str, contract: str, policy: str, repetition: int) -> str:
    return (
        f"{NAMESPACE}--{actor}--{task}--{contract}--{policy}--r{repetition:02d}"
    )


def jobs_for_actor(actor: str) -> list[dict[str, Any]]:
    if actor not in ACTORS:
        raise ValueError(f"unknown actor: {actor}")
    jobs = [
        {
            "actor": actor,
            "task": task.task_id,
            "contract": contract,
            "policy": policy,
            "repetition": repetition,
            "job_id": job_id(
                actor, task.task_id, contract, policy, repetition
            ),
        }
        for task in TASKS
        for contract in CONTRACTS
        for policy in POLICIES
        for repetition in range(REPETITIONS)
    ]
    return sorted(
        jobs,
        key=lambda row: hashlib.sha256(
            f"{NAMESPACE}|{job_key(row['actor'], row['task'], row['contract'], row['policy'], row['repetition'])}".encode(
                "utf-8"
            )
        ).hexdigest(),
    )


def code_paths() -> list[Path]:
    package = Path(__file__).resolve().parents[1]
    return sorted(package.rglob("*.py")) + [PROJECT / "scripts/android_atomic_closed_loop.py"]


def build_manifest() -> dict[str, Any]:
    paths = code_paths()
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError(f"manifest code inputs missing: {missing}")
    if (
        not PINNED_A11Y_APK.is_file()
        or sha256(PINNED_A11Y_APK) != PINNED_A11Y_SHA256
    ):
        raise RuntimeError("pinned accessibility-forwarder APK changed")
    committed_inputs = {rel(path): committed(path) for path in paths}
    if not all(committed_inputs.values()) or not git_clean():
        raise RuntimeError(f"manifest requires clean committed inputs: {committed_inputs}")
    sources = {
        task_id: validate_source(row)
        for task_id, row in source_rows().items()
    }
    checkpoint = load_json(CHECKPOINT_ATTESTATIONS)
    tasks = []
    for spec in TASKS:
        source = sources[spec.task_id]
        if source["params"] != spec.params:
            raise RuntimeError(f"task parameter mismatch: {spec.task_id}")
        tasks.append(
            {
                "task_id": spec.task_id,
                "params": spec.params,
                "app_name": spec.app_name,
                "setup_prefix_labels_scoring_only": list(spec.setup_prefix_labels),
                "terminal_label_scoring_only": spec.terminal_label_scoring_only,
                "source": source,
            }
        )
    jobs = {actor: jobs_for_actor(actor) for actor in ACTORS}
    manifest = {
        "title": "Prospective Native-Android Atomic Closed-Loop Manifest",
        "manifest_version": MANIFEST_VERSION,
        "runner_version": RUNNER_VERSION,
        "paper_evidence": False,
        "target_outcome_blind": True,
        "namespace": NAMESPACE,
        "created_from_git_commit": git_head(),
        "created_from_git_tree": git_tree(),
        "external_root": str(DEFAULT_ROOT),
        "bindings": {
            "code": {rel(path): sha256(path) for path in paths},
            "source_registry": {
                "path": str(SOURCE_REGISTRY),
                "sha256": sha256(SOURCE_REGISTRY),
            },
            "checkpoint_attestations": {
                "path": rel(CHECKPOINT_ATTESTATIONS),
                "sha256": sha256(CHECKPOINT_ATTESTATIONS),
                "models": checkpoint["models"],
            },
            "runtime_assets": {
                "accessibility_forwarder_apk": {
                    "path": str(PINNED_A11Y_APK),
                    "bytes": PINNED_A11Y_APK.stat().st_size,
                    "sha256": sha256(PINNED_A11Y_APK),
                }
            },
        },
        "design": {
            "actors": list(ACTORS),
            "contracts": list(CONTRACTS),
            "policies": list(POLICIES),
            "repetitions": REPETITIONS,
            "tasks": tasks,
            "planned_jobs": jobs,
            "planned_jobs_per_actor": 560,
            "planned_jobs_total": 1120,
            "ordering": f"ascending SHA-256 of {NAMESPACE}|complete-job-key",
        },
        "standard_registration": {
            "frontend": "DAFC SIFT mutual-nearest-neighbor ratio matches",
            "model": "homography",
            "method": "OpenCV RANSAC",
            "rng_seed_reset_per_fit": STANDARD_RNG_SEED,
            "reprojection_threshold_px": STANDARD_RANSAC_THRESHOLD_PX,
            "maximum_iterations": STANDARD_MAX_ITERS,
            "confidence": STANDARD_CONFIDENCE,
            "minimum_inliers": STANDARD_MIN_INLIERS,
            "minimum_inlier_ratio": STANDARD_MIN_INLIER_RATIO,
            "target_boundary_margin_px": TARGET_BOUNDARY_MARGIN_PX,
        },
        "information_boundary": {
            "setup_accessibility_scoring_only": True,
            "authorization_inputs": ["source receipt", "source RGB", "target RGB", "source point"],
            "actor_inputs": ["target RGB", "task instruction"],
            "forbidden_decision_inputs": [
                "target accessibility",
                "target label or box",
                "target reward",
                "another policy action",
            ],
            "official_reward_read_after_exactly_one_target_action": True,
        },
        "failure_policy": {
            "maximum_attempts": 2,
            "retry_only_before_target_observation_model_authorization_or_action": True,
            "outcome_bearing_attempt_never_retried": True,
            "all_attempts_retained": True,
        },
        "analysis": {
            "bootstrap_replicates": 20_000,
            "bootstrap_seed": 588001,
            "cluster_unit": "task",
            "primary_comparison": "dafc minus always separately per actor",
            "noninferiority_margin": -0.05,
            "minimum_target_call_reduction": 0.20,
            "performance_is_not_an_integrity_gate": True,
        },
        "operational": {
            "remote_gpu_index": 0,
            "lifecycle_profile": "android_atomic_gpu0",
            "gpu0_minimum_free_memory_mib": 22_000,
            "gpu0_maximum_utilization_percent": 10,
            "maximum_preexisting_context_memory_mib": 1_024,
            "global_kvm_lock": "/tmp/transitionguard-androidworld-kvm-global.lock",
            "password_persistence": False,
        },
    }
    manifest["jobs_sha256"] = canonical_sha256(jobs)
    manifest["sources_sha256"] = canonical_sha256(tasks)
    criteria = {
        "exact_seven_tasks": len(tasks) == 7,
        "exact_five_successful_and_two_failed_sources": sum(
            int(row["source"]["source_success"]) for row in tasks
        )
        == 5,
        "all_failed_sources_retained_in_jobs": all(
            any(job["task"] == row["task_id"] for job in jobs["qwen25vl"])
            for row in tasks
            if not row["source"]["source_success"]
        ),
        "exact_560_jobs_per_actor": all(len(rows) == 560 for rows in jobs.values()),
        "exact_1120_unique_jobs": len(
            {row["job_id"] for rows in jobs.values() for row in rows}
        )
        == 1120,
        "all_code_inputs_committed": all(committed_inputs.values()),
        "repository_clean": git_clean(),
        "zero_model_or_environment_calls": True,
        "performance_not_used_as_gate": True,
    }
    manifest["criteria"] = criteria
    manifest["pass"] = all(criteria.values())
    return manifest






def command_make_manifest(
    output: Path, signature_json: Path, signature_report: Path
) -> int:
    for candidate in (output, signature_json, signature_report):
        if candidate.exists() or candidate.is_symlink():
            raise RuntimeError(f"refusing to overwrite manifest-stage output: {candidate}")
    manifest = build_manifest()
    write_new_json(output, manifest)
    signature = {
        "title": "C586c Native-Android Atomic V3 Manifest Signature",
        "paper_evidence": False,
        "manifest": rel(output),
        "manifest_sha256": sha256(output),
        "manifest_version": manifest["manifest_version"],
        "created_from_git_commit": manifest["created_from_git_commit"],
        "created_from_git_tree": manifest["created_from_git_tree"],
        "jobs_sha256": manifest["jobs_sha256"],
        "sources_sha256": manifest["sources_sha256"],
        "planned_jobs": manifest["design"]["planned_jobs_total"],
        "successful_source_receipts": 5,
        "failed_source_receipts_retained": 2,
        "zero_model_requests": True,
        "zero_environment_actions": True,
        "pass": manifest["pass"] is True,
    }
    write_new_json(signature_json, signature)
    write_new_text(
        signature_report,
        "\n".join(
            [
                "# C586c Native-Android Atomic V3 Manifest Signature",
                "",
                "The C585 protocol, result-blind GPU-0 amendment, and pinned runtime-asset repair are frozen before any V3 model request or target action.",
                "",
                f"- manifest SHA-256: `{signature['manifest_sha256']}`",
                f"- jobs SHA-256: `{signature['jobs_sha256']}`",
                f"- sources SHA-256: `{signature['sources_sha256']}`",
                "- population: all seven sealed source tasks; five successful receipts and two failed receipts retained",
                "- planned target jobs: 1,120",
                "- model requests and environment actions at freeze: zero",
                "",
            ]
        ),
    )
    print(
        json.dumps(
            {
                "path": str(output),
                "sha256": sha256(output),
                "signature_json": str(signature_json),
                "signature_report": str(signature_report),
                "jobs_sha256": manifest["jobs_sha256"],
                "criteria": manifest["criteria"],
                "pass": manifest["pass"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if manifest["pass"] else 1


def verify_manifest(path: Path) -> dict[str, Any]:
    if not committed(path) or not git_clean():
        raise RuntimeError("runtime requires a committed manifest and clean repository")
    manifest = load_json(path)
    if (
        manifest.get("manifest_version") != MANIFEST_VERSION
        or manifest.get("runner_version") != RUNNER_VERSION
        or manifest.get("namespace") != NAMESPACE
        or manifest.get("pass") is not True
        or manifest.get("external_root") != str(DEFAULT_ROOT)
    ):
        raise RuntimeError("manifest identity changed")
    source_commit = manifest.get("created_from_git_commit")
    source_tree = manifest.get("created_from_git_tree")
    if (
        not isinstance(source_commit, str)
        or not isinstance(source_tree, str)
        or git_tree(source_commit) != source_tree
    ):
        raise RuntimeError("manifest source commit/tree binding changed")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", source_commit, "HEAD"],
        cwd=PROJECT,
        check=False,
    )
    changed = set(
        subprocess.check_output(
            ["git", "diff", "--name-only", source_commit, "HEAD"],
            cwd=PROJECT,
            text=True,
        ).splitlines()
    )
    permitted_after_freeze = {
        rel(path),
        "reports/c586c_android_atomic_manifest_v3_signature.md",
        "validation/c586c_android_atomic_manifest_v3_signature.json",
    }
    if ancestor.returncode != 0 or not changed.issubset(permitted_after_freeze):
        raise RuntimeError(f"repository changed beyond the frozen manifest: {changed}")
    for name, expected in manifest["bindings"]["code"].items():
        candidate = PROJECT / name
        if not candidate.is_file() or sha256(candidate) != expected:
            raise RuntimeError(f"bound code changed: {name}")
    source_binding = manifest["bindings"]["source_registry"]
    if sha256(Path(source_binding["path"])) != source_binding["sha256"]:
        raise RuntimeError("source registry changed")
    runtime_binding = manifest["bindings"]["runtime_assets"][
        "accessibility_forwarder_apk"
    ]
    runtime_asset = Path(runtime_binding["path"])
    if (
        runtime_asset != PINNED_A11Y_APK
        or not runtime_asset.is_file()
        or runtime_asset.stat().st_size != runtime_binding["bytes"]
        or sha256(runtime_asset) != runtime_binding["sha256"]
        or runtime_binding["sha256"] != PINNED_A11Y_SHA256
    ):
        raise RuntimeError("bound accessibility-forwarder APK changed")
    jobs = manifest["design"]["planned_jobs"]
    if canonical_sha256(jobs) != manifest["jobs_sha256"]:
        raise RuntimeError("manifest job plan changed")
    if canonical_sha256(manifest["design"]["tasks"]) != manifest["sources_sha256"]:
        raise RuntimeError("manifest source plan changed")
    return manifest














from dafc.models.responses import (
    target_instruction as target_instruction,
    compact_model_response as compact_model_response,
    normalized_to_pixel as normalized_to_pixel,
)






def execute_click(env: Any, point: list[float], image: np.ndarray) -> list[float]:
    if not point_in_bounds(point, image, 0.0):
        raise RuntimeError("target point is outside the observation")
    from android_world.env import json_action

    height, width = image.shape[:2]
    executed = [
        min(max(int(round(point[0])), 0), width - 1),
        min(max(int(round(point[1])), 0), height - 1),
    ]
    env.execute_action(
        json_action.JSONAction(action_type="click", x=executed[0], y=executed[1])
    )
    time.sleep(POST_ACTION_SETTLE_S)
    return [float(executed[0]), float(executed[1])]


def ensure_clean_observation_ui(env: Any, stage: str) -> dict[str, Any]:
    before = C482.adb("shell", "settings", "get", "system", "pointer_location")
    env.hide_automation_ui()
    time.sleep(0.25)
    after = C482.adb("shell", "settings", "get", "system", "pointer_location")
    if after != "0":
        raise RuntimeError(f"pointer overlay remains enabled at {stage}: {after!r}")
    return {"stage": stage, "before": before, "after": after, "clean": True}


def setup_target_state(env: Any, task_registry: dict[str, Any], spec: AtomicTask, contract_id: str):
    from android_world.env import json_action
    from android_world.env.setup_device import apps as setup_apps

    contracts = {row.contract_id: row for row in C482.CONTRACTS}
    contract = contracts[contract_id]
    C482.apply_contract(contract)
    env.reset(go_home=True)
    if spec.task_id == "AudioRecorderRecordAudio":
        setup_apps.AudioRecorder.setup(env)
    task = task_registry[spec.task_id](dict(spec.params))
    try:
        task.initialize_task(env)
        ui_receipts = [ensure_clean_observation_ui(env, "after_task_initialization")]
        initial_reward = float(task.is_successful(env))
        if initial_reward != 0.0:
            raise RuntimeError("task is already successful before setup")
        env.execute_action(json_action.JSONAction(action_type="open_app", app_name=spec.app_name))
        time.sleep(6.0)
        ui_receipts.append(ensure_clean_observation_ui(env, "after_open_app"))
        setup_actions = []
        for index, label in enumerate(spec.setup_prefix_labels):
            state = C482.wait_state(env, required_label=label, timeout_s=12.0)
            action = C482.click_label(env, state, label)
            reward = float(task.is_successful(env))
            setup_actions.append({"index": index, **action, "official_reward": reward})
            if reward != 0.0:
                raise RuntimeError("setup prefix completed the task before target action")
            ui_receipts.append(ensure_clean_observation_ui(env, f"after_setup_{index:02d}"))
        state = C482.wait_state(
            env,
            required_label=spec.terminal_label_scoring_only,
            timeout_s=12.0,
        )
        raw = C482.pixels(state)
        observed = C482.observe(raw, contract)
        reward = float(task.is_successful(env))
        if reward != 0.0:
            raise RuntimeError("target state is successful before target action")
        return task, str(task.goal), raw, observed, setup_actions, ui_receipts
    except BaseException:
        try:
            task.tear_down(env)
        except Exception:
            pass
        raise


def task_source(manifest: dict[str, Any], task_id: str) -> dict[str, Any]:
    rows = {
        row["task_id"]: row["source"] for row in manifest["design"]["tasks"]
    }
    return rows[task_id]


def result_path(root: Path, actor: str, job: dict[str, Any], attempt: int) -> Path:
    return root / "results" / actor / f"{job['job_id']}--a{attempt:02d}.json"


def append_ledger(root: Path, actor: str, row: dict[str, Any]) -> None:
    path = root / "ledger" / f"{actor}.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_CLOEXEC, 0o600
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        os.write(
            descriptor,
            (json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n").encode(
                "utf-8"
            ),
        )
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
) -> dict[str, Any]:
    spec = TASK_BY_ID[job["task"]]
    source = task_source(manifest, spec.task_id)
    path = result_path(root, actor, job, attempt)
    image_dir = root / "images" / actor / f"{job['job_id']}--a{attempt:02d}"
    stage = "start"
    task = None
    model_calls = 0
    authorization_started = False
    target_action_started = False
    started = time.perf_counter()
    try:
        stage = "target_setup"
        task, goal, raw, observed, setup_actions, ui_receipts = setup_target_state(
            env, task_registry, spec, job["contract"]
        )
        stage = "target_observation"
        before = image_receipt(image_dir / "before.png", observed)
        raw_before = image_receipt(image_dir / "before_raw.png", raw)
        stage = "authorization"
        authorization_started = job["policy"] != "always"
        authorization = authorize(job["policy"], source, observed)
        model_response = None
        requested_action = authorization.get("mapped_action_xy")
        action_source = "source_reuse" if authorization["accepted"] else "target_actor"
        if not authorization["accepted"]:
            stage = "model_request"
            request_id = f"{job['job_id']}--a{attempt:02d}"
            model_calls = 1
            response = GROUND_CLIENT.infer(
                service_url,
                request_id,
                Path(before["path"]),
                instruction=target_instruction(goal),
                timeout_s=240.0,
            )
            model_response = compact_model_response(response)
            requested_action = normalized_to_pixel(
                response.get("result", {}).get("parsed_action", {}), observed
            )
        if requested_action is None:
            stage = "scientific_parse_failure"
            result = {
                "status": "complete",
                "scientific_failure": "target_actor_action_parse_failure",
                "success": False,
                "official_reward": 0.0,
                "action_executed": False,
                "executed_action_xy": None,
            }
            after = None
        else:
            stage = "target_action"
            target_action_started = True
            executed_action = execute_click(env, requested_action, raw)
            stage = "reward_read"
            ui_receipts.append(ensure_clean_observation_ui(env, "after_target_action"))
            next_state = env.get_state(wait_to_stabilize=True)
            next_raw = C482.pixels(next_state)
            next_observed = C482.observe(
                next_raw,
                {row.contract_id: row for row in C482.CONTRACTS}[job["contract"]],
            )
            after = image_receipt(image_dir / "after.png", next_observed)
            reward = float(task.is_successful(env))
            result = {
                "status": "complete",
                "scientific_failure": None if reward == 1.0 else "official_reward_zero",
                "success": reward == 1.0,
                "official_reward": reward,
                "action_executed": True,
                "executed_action_xy": executed_action,
            }
        payload = {
            "title": "Native-Android Atomic Closed-Loop Job",
            "paper_evidence": True,
            "runner_version": RUNNER_VERSION,
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_path),
            "jobs_sha256": manifest["jobs_sha256"],
            "job": job,
            "attempt": attempt,
            "source": source,
            "goal": goal,
            "setup": {
                "accessibility_scoring_only": True,
                "actions": setup_actions,
                "reward_remained_zero": all(
                    row["official_reward"] == 0.0 for row in setup_actions
                ),
            },
            "observation_ui": {
                "all_clean": all(row["clean"] for row in ui_receipts),
                "receipts": ui_receipts,
            },
            "before": before,
            "before_raw": raw_before,
            "authorization": authorization,
            "action_source": action_source,
            "model_calls": model_calls,
            "model_response": model_response,
            "requested_action_xy": requested_action,
            "after": after,
            **result,
            "outcome_bearing": True,
            "retry_permitted": False,
            "elapsed_s": time.perf_counter() - started,
        }
    except Exception as caught:
        outcome_bearing = bool(
            stage in {"target_observation", "authorization", "model_request", "target_action", "reward_read"}
            or model_calls
            or authorization_started
            or target_action_started
        )
        payload = {
            "title": "Native-Android Atomic Closed-Loop Job Failure",
            "paper_evidence": True,
            "runner_version": RUNNER_VERSION,
            "manifest": str(manifest_path.resolve()),
            "manifest_sha256": sha256(manifest_path),
            "jobs_sha256": manifest["jobs_sha256"],
            "job": job,
            "attempt": attempt,
            "status": "outcome_failure" if outcome_bearing else "infrastructure_failure",
            "stage": stage,
            "error_type": type(caught).__name__,
            "error": str(caught),
            "model_calls": model_calls,
            "authorization_started": authorization_started,
            "target_action_started": target_action_started,
            "outcome_bearing": outcome_bearing,
            "retry_permitted": not outcome_bearing and attempt == 0,
            "success": False if outcome_bearing else None,
            "elapsed_s": time.perf_counter() - started,
        }
    finally:
        if task is not None:
            try:
                task.tear_down(env)
            except Exception:
                pass
        try:
            C482.restore_display()
        except Exception:
            pass
    write_new_json(path, payload)
    append_ledger(
        root,
        actor,
        {
            "attempt_path": str(path.resolve()),
            "attempt_sha256": sha256(path),
            "job_id": job["job_id"],
            "attempt": attempt,
            "status": payload["status"],
            "outcome_bearing": payload["outcome_bearing"],
            "retry_permitted": payload["retry_permitted"],
        },
    )
    return payload


def existing_attempts(root: Path, actor: str, job: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for attempt in (0, 1):
        path = result_path(root, actor, job, attempt)
        if path.exists():
            payload = load_json(path)
            rows.append(payload)
    return rows


def next_attempt(root: Path, actor: str, job: dict[str, Any]) -> int | None:
    rows = existing_attempts(root, actor, job)
    if any(row.get("outcome_bearing") is True for row in rows):
        return None
    if not rows:
        return 0
    if len(rows) == 1 and rows[0].get("retry_permitted") is True:
        return 1
    return None


def command_run_batch(
    manifest_path: Path, actor: str, service_url: str, root: Path
) -> int:
    if root.resolve() != DEFAULT_ROOT.resolve():
        raise RuntimeError("external result root changed")
    manifest = verify_manifest(manifest_path)
    if actor not in ACTORS:
        raise RuntimeError("actor changed")
    health = GROUND_CLIENT.health(service_url, timeout_s=15.0)
    if health.get("kind") != actor:
        raise RuntimeError(f"service actor mismatch: {health}")
    root.mkdir(parents=True, exist_ok=True)
    activation = root / "activation.json"
    if not activation.exists():
        write_new_json(
            activation,
            {
                "manifest": str(manifest_path.resolve()),
                "manifest_sha256": sha256(manifest_path),
                "jobs_sha256": manifest["jobs_sha256"],
                "namespace": NAMESPACE,
                "runner_version": RUNNER_VERSION,
            },
        )
    else:
        existing = load_json(activation)
        if existing.get("manifest_sha256") != sha256(manifest_path):
            raise RuntimeError("external activation belongs to another manifest")

    from android_env.wrappers import a11y_grpc_wrapper
    from android_world import registry
    from android_world.env import env_launcher

    a11y_grpc_wrapper._get_accessibility_forwarder_apk = PINNED_A11Y_APK.read_bytes
    task_registry = registry.TaskRegistry().get_registry(
        registry.TaskRegistry.ANDROID_WORLD_FAMILY
    )
    env = env_launcher.load_and_setup_env(
        console_port=5554,
        emulator_setup=False,
        freeze_datetime=True,
        adb_path=str(C459.ADB),
        grpc_port=8554,
    )
    try:
        completed = 0
        for index, job in enumerate(manifest["design"]["planned_jobs"][actor], 1):
            attempt = next_attempt(root, actor, job)
            while attempt is not None:
                payload = run_one(
                    env=env,
                    task_registry=task_registry,
                    manifest=manifest,
                    manifest_path=manifest_path,
                    actor=actor,
                    service_url=service_url,
                    root=root,
                    job=job,
                    attempt=attempt,
                )
                print(
                    json.dumps(
                        {
                            "progress": f"{index}/560",
                            "job_id": job["job_id"],
                            "attempt": attempt,
                            "status": payload["status"],
                            "success": payload.get("success"),
                            "model_calls": payload.get("model_calls"),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                attempt = next_attempt(root, actor, job)
            if any(
                row.get("outcome_bearing") is True
                for row in existing_attempts(root, actor, job)
            ):
                completed += 1
        return 0 if completed == 560 else 2
    finally:
        C482.restore_display()
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    manifest = subparsers.add_parser("make-manifest")
    manifest.add_argument("--output", type=Path, default=DEFAULT_MANIFEST)
    manifest.add_argument(
        "--signature-json", type=Path, default=DEFAULT_SIGNATURE_JSON
    )
    manifest.add_argument(
        "--signature-report", type=Path, default=DEFAULT_SIGNATURE_REPORT
    )
    run = subparsers.add_parser("run-batch")
    run.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    run.add_argument("--actor", choices=ACTORS, required=True)
    run.add_argument("--service-url", required=True)
    run.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "make-manifest":
        return command_make_manifest(
            args.output, args.signature_json, args.signature_report
        )
    return command_run_batch(args.manifest, args.actor, args.service_url, args.root)


if __name__ == "__main__":
    raise SystemExit(main())
