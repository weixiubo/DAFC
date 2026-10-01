#!/usr/bin/env python3
"""Development audit for the formal-preflight Android live runner boundary."""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from android_env.wrappers import a11y_grpc_wrapper
from android_world import registry
from android_world.env import env_launcher

import dafc.adapters.android.c459_androidworld_clock_mechanics_audit as C459
import dafc.adapters.android.c482_androidworld_v2_breadth_audit as C482
import dafc.adapters.android.live_androidworld_transition_guard as LIVE
from dafc.models.lightglue_runtime_service import LightGlueServiceClient


from dafc.paths import PROJECT
ROOT = PROJECT / "tmp" / "c488_android_live_runner_integrity_audit"
OUTPUT = PROJECT / "validation" / "c488_android_live_runner_integrity_audit.json"
C487_INPUT = PROJECT / "validation" / "c487_androidworld_parameterized_timer_cohort_audit.json"
DEVELOPMENT_INSTANCE_ID = "ClockTimerEntryDev02"


def compact_episode(episode: dict[str, Any]) -> dict[str, Any]:
    return {
        key: episode[key]
        for key in (
            "runner_version",
            "method_version",
            "episode_id",
            "task_id",
            "params",
            "parameter_fingerprint",
            "goal",
            "contract",
            "policy",
            "success",
            "failure_reason",
            "executed_actions",
            "model_calls",
            "reused_actions",
            "postcondition_failures",
            "confirmed_transition_ids",
            "invalidated_transition_ids",
            "replay_circuit_open",
        )
    }


def write_episode(name: str, episode: dict[str, Any]) -> Path:
    path = ROOT / f"{name}.json"
    path.write_text(
        json.dumps(episode, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return path


def mock_response(
    request_id: str,
    before_path: Path,
    point: tuple[float, float],
    instruction: str,
) -> dict[str, Any]:
    image = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not load mock input {before_path}")
    height, width = image.shape[:2]
    normalized = [point[0] / width, point[1] / height]
    return {
        "service_version": "c488-development-coordinate-mock-v1",
        "kind": "qwen25vl",
        "request_sha256": f"development-{request_id}",
        "service_s": 0.0,
        "result": {
            "id": request_id,
            "image_sha256": LIVE.sha256(before_path),
            "image_width": width,
            "image_height": height,
            "input_tokens": 0,
            "generated_tokens": 0,
            "inference_s": 0.0,
            "parsed_action": {
                "valid": True,
                "normalized_xy": normalized,
            },
            "raw_response": "development coordinate mock",
            "instruction": instruction,
        },
    }


def ablation_prefix_unit_check() -> dict[str, Any]:
    guard = LIVE.GUARD
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    memory = [
        guard.Transition("predecessor", 0, True, image, (2.0, 2.0), image),
        guard.Transition(
            "terminal", 1, True, image, (8.0, 8.0), image, terminal=True
        ),
    ]
    original_source_effect = guard.source_effect_receipt
    original_authorize = guard.authorize

    def observable(_transition):
        return {"observable": True, "accepted": False, "reason": "unit"}

    def authorize(transition, current, **_kwargs):
        if transition.transition_id == "predecessor":
            return guard.reject_receipt(
                transition, "G", "unit_predecessor_geometry_failure"
            ), None
        receipt = {
            "method_version": guard.METHOD_VERSION,
            "transition_id": transition.transition_id,
            "demonstration_order": transition.demonstration_order,
            "authorized": True,
            "decision": "reuse",
            "failed_gate": None,
            "reason": "unit_terminal_authorized",
            "source_success": True,
            "geometry": {},
            "precondition": {},
            "mapped_action_xy": [8.0, 8.0],
        }
        context = guard.AuthorizationContext(
            transition_id="terminal",
            demonstration_order=1,
            mapped_action_xy=(8.0, 8.0),
            warped_before=current.copy(),
            warped_after=current.copy(),
            authorization_receipt=receipt,
        )
        return receipt, context

    try:
        guard.source_effect_receipt = observable
        guard.authorize = authorize
        sg, _ = guard.select(memory, image, require_precondition=False)
        sgp, _ = guard.select(memory, image, require_precondition=True)
    finally:
        guard.source_effect_receipt = original_source_effect
        guard.authorize = original_authorize
    return {
        "sg_selects_terminal_without_prefix_rule": (
            sg["decision"] == "reuse"
            and sg["selected_transition_id"] == "terminal"
        ),
        "sgp_retains_terminal_prefix_rule": (
            sgp["decision"] == "fallback"
            and any(
                row.get("reason") == "terminal_transition_requires_confirmed_prefix"
                for row in sgp["evaluations"]
            )
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--lightglue-url", default="http://127.0.0.1:48242"
    )
    args = parser.parse_args()
    prior = json.loads(C487_INPUT.read_text(encoding="utf-8"))
    if not prior.get("pass") or prior.get("paper_evidence") is not False:
        raise RuntimeError("C487 development timer audit must pass")
    source_fixture = next(
        row
        for row in prior["source_episodes"]
        if row["task_id"] == DEVELOPMENT_INSTANCE_ID
    )
    params = dict(source_fixture["params"])
    scoring_actions = source_fixture["actions_scoring_only"]
    requested_points = [
        (
            float(row["coordinate_scoring_only"][0]) + 0.4,
            float(row["coordinate_scoring_only"][1]) + 0.4,
        )
        for row in scoring_actions
    ]
    ROOT.mkdir(parents=True, exist_ok=True)
    lightglue = LightGlueServiceClient(args.lightglue_url)
    health = lightglue.health()
    if not health.get("ready"):
        raise RuntimeError("LightGlue service is not ready")

    source_neutral_history = LIVE.history_text(
        [{"action_xy": [123.0, 456.0], "source": "base_agent"}]
    ) == LIVE.history_text(
        [{"action_xy": [123.0, 456.0], "source": "transition_guard"}]
    )
    resolved = LIVE.resolved_task_spec("ClockTimerEntry", params)
    different_fingerprint = LIVE.parameter_fingerprint(params) != LIVE.parameter_fingerprint(
        {**params, "seconds": (params["seconds"] + 1) % 60}
    )
    invalid_timer_rejected = False
    try:
        LIVE.validated_timer_params({"hours": 24, "minutes": 0, "seconds": 1})
    except ValueError:
        invalid_timer_rejected = True

    a11y_grpc_wrapper._get_accessibility_forwarder_apk = C459.A11Y_APK.read_bytes
    task_registry = registry.TaskRegistry().get_registry(
        registry.TaskRegistry.ANDROID_WORLD_FAMILY
    )
    env = None
    source = None
    replay_episodes: dict[str, dict[str, Any]] = {}
    original_infer = LIVE.GROUND_CLIENT.infer
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
        cursor = 0

        def coordinate_mock(
            _url: str,
            request_id: str,
            before_path: Path,
            *,
            instruction: str,
        ) -> dict[str, Any]:
            nonlocal cursor
            if cursor >= len(requested_points):
                raise RuntimeError("source requested more mock actions than frozen")
            point = requested_points[cursor]
            cursor += 1
            return mock_response(request_id, before_path, point, instruction)

        LIVE.GROUND_CLIENT.infer = coordinate_mock
        source = LIVE.run_episode(
            env=env,
            task_registry=task_registry,
            task_id="ClockTimerEntry",
            contract_id="identity",
            policy="always",
            kind="qwen25vl",
            model_service_url="http://development.invalid",
            run_id="c488-development-source",
            attempt=0,
            output_root=ROOT / "images",
            timer_params=params,
        )
        if cursor != len(requested_points):
            raise RuntimeError("source did not consume the frozen mock sequence")
        source_path = write_episode("source", source)
        memory_goal, memory_params, memory = LIVE.load_memory(source_path)

        def forbid_model_call(*_args, **_kwargs):
            raise RuntimeError("qualified replay unexpectedly called the model")

        LIVE.GROUND_CLIENT.infer = forbid_model_call
        for policy in ("sg", "sgp", "full"):
            episode = LIVE.run_episode(
                env=env,
                task_registry=task_registry,
                task_id="ClockTimerEntry",
                contract_id="identity",
                policy=policy,
                kind="qwen25vl",
                model_service_url="http://development.invalid",
                run_id=f"c488-development-{policy}",
                attempt=0,
                output_root=ROOT / "images",
                timer_params=params,
                memory_goal=memory_goal,
                memory_params=memory_params,
                memory=memory,
                match_provider=lightglue.correspondences,
            )
            replay_episodes[policy] = episode
            write_episode(policy, episode)
    finally:
        LIVE.GROUND_CLIENT.infer = original_infer
        C482.restore_display()
        if env is not None:
            env.close()

    if source is None:
        raise RuntimeError("source episode was not produced")
    source_action_receipts_exact = all(
        row["action_xy"] == row["decision"]["executed_action_xy"]
        and all(float(value).is_integer() for value in row["action_xy"])
        for row in source["steps"]
    )
    requested_and_executed_are_distinguished = all(
        row["decision"]["requested_action_xy"]
        != row["decision"]["executed_action_xy"]
        for row in source["steps"]
    )
    no_action_source_in_model_history = all(
        "base_agent" not in row["history"]
        and "transition_guard" not in row["history"]
        for row in source["steps"]
    ) and all(
        "base_agent" not in row["history"]
        and "transition_guard" not in row["history"]
        for episode in replay_episodes.values()
        for row in episode["steps"]
    )
    replay_count = len(scoring_actions)
    policy_wiring = {
        "sg_has_no_pre_or_post_gate": any(
            (evaluation.get("precondition") or {}).get("reason")
            == "disabled_by_ablation"
            for row in replay_episodes["sg"]["steps"]
            for evaluation in row["decision"]["evaluations"]
        )
        and all(
            row["post_verification"] is None
            for row in replay_episodes["sg"]["steps"]
        ),
        "sgp_has_pre_but_no_post_gate": all(
            row["post_verification"] is None
            for row in replay_episodes["sgp"]["steps"]
        )
        and any(
            (evaluation.get("precondition") or {}).get("accepted") is True
            for row in replay_episodes["sgp"]["steps"]
            for evaluation in row["decision"]["evaluations"]
        ),
        "full_verifies_every_replay_including_terminal": (
            len(replay_episodes["full"]["steps"]) == replay_count
            and all(
                row["post_verification"] is not None
                and row["post_verification"]["post_verified"]
                for row in replay_episodes["full"]["steps"]
            )
        ),
    }
    prefix_checks = ablation_prefix_unit_check()
    criteria = {
        "development_only": True,
        "oracle_coordinate_mock_disclosed_and_excluded_from_evidence": True,
        "live_task_spec_contains_no_action_labels": not hasattr(
            resolved, "action_labels"
        ),
        "model_history_is_action_source_neutral": source_neutral_history
        and no_action_source_in_model_history,
        "timer_parameters_round_trip_exactly": (
            source["params"] == params
            and all(row["params"] == params for row in replay_episodes.values())
        ),
        "parameter_fingerprint_prevents_instance_collision": different_fingerprint
        and all(
            f"p{source['parameter_fingerprint'][:12]}" in row["episode_id"]
            for row in (source, *replay_episodes.values())
        ),
        "invalid_timer_range_rejected": invalid_timer_rejected,
        "source_receipt_records_exact_executed_integer_pixels": source_action_receipts_exact,
        "requested_subpixel_and_executed_pixel_are_both_auditable": requested_and_executed_are_distinguished,
        "source_officially_successful": source["success"]
        and source["model_calls"] == replay_count,
        "all_three_replay_policies_officially_succeed": all(
            row["success"] for row in replay_episodes.values()
        ),
        "all_three_replay_policies_use_no_model_calls": all(
            row["model_calls"] == 0 for row in replay_episodes.values()
        ),
        "all_three_replay_policies_reuse_every_transition": all(
            row["reused_actions"] == replay_count
            for row in replay_episodes.values()
        ),
        **policy_wiring,
        **prefix_checks,
    }
    result = {
        "title": "C488 Android Live Runner Integrity Audit",
        "paper_evidence": False,
        "purpose": (
            "audit formal-preflight task parameters, model-history parity, "
            "executed coordinates, and SG/SGP/full policy wiring"
        ),
        "development_fixture": {
            "source": "C487 ClockTimerEntryDev02",
            "params": params,
            "point_actions": replay_count,
            "oracle_coordinate_mock": True,
            "formal_namespace_consumed": False,
        },
        "services": {
            "lightglue_ready": bool(health.get("ready")),
            "lightglue_service_version": health.get("service_version"),
        },
        "episodes": {
            "source": compact_episode(source),
            **{
                policy: compact_episode(episode)
                for policy, episode in replay_episodes.items()
            },
        },
        "criteria": criteria,
        "pass": all(criteria.values()),
        "artifacts": {
            name: (ROOT / f"{name}.json").relative_to(PROJECT).as_posix()
            for name in ("source", "sg", "sgp", "full")
        },
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "episodes": result["episodes"],
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
