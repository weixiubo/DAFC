#!/usr/bin/env python3
"""Run one real MiniWoB episode with an online GUI model or TransitionGuard.

Every fallback action is generated from the screenshot actually observed in
the live episode through the persistent V5 grounder service.  The runner never
precomputes actions from an oracle trajectory.  DOM-derived task plans and
boxes are absent; MiniWoB reward is read only after actions for episode scoring
and source-success admission.
"""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np
from miniwob.environment import MiniWoBEnvironment


from dafc.paths import PROJECT
SCRIPTS = PROJECT / "scripts"
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


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
C388 = import_module('dafc.adapters.browser.c388_browser_native_contract_development_pilot')
C400 = import_module('dafc.adapters.browser.c400_final_browser_capture')
C430 = import_module('dafc.adapters.browser.c430_multistep_transition_feasibility')
C432 = import_module('dafc.adapters.browser.c432_multistep_native_contract_capture')
GUARD = import_module('dafc.transitions')
HARNESS = import_module('dafc.adapters.browser.miniwob_agent_harness')
LIGHTGLUE = import_module('dafc.models.lightglue_runtime_frontend')
CLIENT = import_module('dafc.models.client')


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
    matches = [
        row
        for row in C432.multistep_contracts(124)
        if row["name"] == name
    ]
    if len(matches) != 1:
        raise KeyError(f"unknown contract: {name}")
    return matches[0]


def history_text(actions: list[dict[str, Any]]) -> str:
    if not actions:
        return "This is the task's initial state."
    selected = actions[-HISTORY_HORIZON:]
    first_step = len(actions) - len(selected) + 1
    return "\n".join(
        f"Step{first_step + offset}: clicked at full-screenshot pixel "
        f"coordinates ({row['action_image_xy'][0]:.1f},"
        f"{row['action_image_xy'][1]:.1f})."
        for offset, row in enumerate(selected)
    )


def generic_instruction(goal: str, history: str) -> str:
    return (
        f"Original task: {goal}\n"
        f"Previously executed action history:\n{history}\n"
        "Using the current screenshot and action history, select exactly one "
        "next executable click that advances the original task. Do not repeat "
        "a completed action."
    )


def normalized_to_image(
    normalized_xy: list[float], image: dict[str, Any]
) -> tuple[tuple[float, float], bool]:
    width = float(image["width"])
    height = float(image["height"])
    raw = (
        float(normalized_xy[0]) * width,
        float(normalized_xy[1]) * height,
    )
    clamped = (
        min(max(raw[0], 0.0), max(width - 1.0, 0.0)),
        min(max(raw[1], 0.0), max(height - 1.0, 0.0)),
    )
    return clamped, clamped != raw


def sequential_replay_decision(
    policy: str,
    memory: list[Any],
    cursor: int,
    current: np.ndarray,
) -> tuple[dict[str, Any], tuple[float, float] | None]:
    """Select the next trajectory action without inspecting current pixels."""

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


def image_to_css(
    action_image: tuple[float, float], contract: dict[str, Any]
) -> list[float]:
    scale = float(contract["capture_scale"])
    return [action_image[0] / scale, action_image[1] / scale]


def load_memory(path: Path) -> tuple[str, list[Any]]:
    episode = json.loads(path.read_text(encoding="utf-8"))
    if episode.get("runner_version") != RUNNER_VERSION:
        raise RuntimeError("source episode runner version mismatch")
    if episode.get("contract") != "source":
        raise RuntimeError("memory source must use the source contract")
    if episode.get("policy") != "always":
        raise RuntimeError("memory source must use the always-call policy")
    if not episode.get("success"):
        raise RuntimeError("memory source episode did not succeed")
    transitions = []
    source_rows = episode["steps"]
    for row in source_rows:
        if not row.get("action_executed"):
            raise RuntimeError("successful source contains a step without action")
        before_path = PROJECT / row["before"]["path"]
        after_path = PROJECT / row["after"]["path"]
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
                action_xy=tuple(float(value) for value in row["action_image_xy"]),
                after=after,
                terminal=int(row["step"]) == len(source_rows) - 1,
            )
        )
    return str(episode["goal"]), transitions


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
    service_url: str,
    kind: str,
    request_id: str,
    image_path: Path,
    goal: str,
    history: str,
) -> dict[str, Any]:
    if kind == "har":
        return CLIENT.infer(
            service_url,
            request_id,
            image_path,
            goal=goal,
            history=history,
        )
    return CLIENT.infer(
        service_url,
        request_id,
        image_path,
        instruction=generic_instruction(goal, history),
    )


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


def run_episode(
    *,
    task: str,
    seed: int,
    contract: dict[str, Any],
    policy: str,
    kind: str,
    output_dir: Path,
    grounder: Callable[[str, Path, str, str], dict[str, Any]],
    run_id: str = "development-legacy",
    attempt: int = 0,
    memory_goal: str | None = None,
    memory: list[Any] | None = None,
    match_provider: Callable[[np.ndarray, np.ndarray], dict[str, Any]] | None = None,
    geometry_frontend: str = "classical_sift",
    executor: Callable[
        [MiniWoBEnvironment, list[float], str, int],
        tuple[dict[str, Any] | None, dict[str, Any]],
    ]
    | None = None,
    risk_certificate: Any | None = None,
    require_risk_certificate: bool = False,
    memory_enable_after_steps: int = 0,
    score_postcondition_for_ablations: bool = False,
) -> dict[str, Any]:
    if task not in TASKS:
        raise ValueError(f"task is outside the frozen eight-task cohort: {task}")
    if policy not in {"always", *SEQUENTIAL_POLICIES, *GUARDED_POLICIES}:
        raise ValueError(f"unknown policy: {policy}")
    if policy != "always" and (memory_goal is None or memory is None):
        raise ValueError("replay policies require a successful source memory")
    if require_risk_certificate and policy != "full":
        raise ValueError("risk-limited activation applies only to full policy")
    if score_postcondition_for_ablations and policy not in GUARDED_POLICIES:
        raise ValueError(
            "scoring-only postconditions require a guarded replay policy"
        )
    if memory_enable_after_steps < 0:
        raise ValueError("memory_enable_after_steps must be non-negative")
    if policy == "always" and memory_enable_after_steps != 0:
        raise ValueError("always policy cannot have a memory warm-start prefix")
    if attempt < 0:
        raise ValueError("attempt must be non-negative")
    if (
        not run_id
        or len(run_id) > 80
        or any(not (ch.isalnum() or ch in "-_.") for ch in run_id)
    ):
        raise ValueError("run_id must be a 1-80 character safe path component")

    episode_id = (
        f"{run_id}--{task}--{seed}--{contract['name']}--{kind}--{policy}--"
        f"a{attempt:02d}"
    )
    episode_dir = output_dir / episode_id
    env = MiniWoBEnvironment(subdomain=task, action_space_config="all_supported")
    started = time.perf_counter()
    try:
        env.instance.driver.set_window_size(900, 1000)
        observation, _ = HARNESS.reset_agent_episode(env, seed)
        C400.reset_harness(env.instance.driver)
        goal = str(observation["utterance"])
        if memory_goal is not None and goal != memory_goal:
            raise RuntimeError("exact-goal memory retrieval mismatch")
        if contract.get("script"):
            env.instance.driver.execute_script(contract["script"])

        suffix = ".jpg" if contract["format"] == "jpeg" else ".png"
        action_history: list[dict[str, Any]] = []
        steps = []
        model_calls = 0
        reused_actions = 0
        invalidated_ids: set[str] = set()
        confirmed_ids: set[str] = set()
        progress_recovered_ids: set[str] = set()
        consumed_sequential_ids: list[str] = []
        sequential_cursor = 0
        replay_circuit_open = False
        terminal_outcome = None
        failure_reason = None
        execute_action = executor or (
            lambda active_env, point, _source, _step: C430.click(active_env, point)
        )

        for step in range(MAX_STEPS):
            before_path = episode_dir / f"step_{step:02d}_before{suffix}"
            before = C388.capture(env.instance.driver, before_path, contract)
            current = cv2.imread(str(before_path), cv2.IMREAD_COLOR)
            if current is None:
                raise RuntimeError(f"could not load captured image: {before_path}")

            post_receipt = None
            scoring_post_receipt = None
            guard_select_s = 0.0
            post_verify_s = 0.0
            scoring_only_post_verify_s = 0.0
            history = history_text(action_history)

            decision: dict[str, Any]
            service_response = None
            action_image: tuple[float, float] | None = None
            action_source = None
            guard_context = None
            sequential_id = None
            available_memory = [
                row
                for row in (memory or [])
                if row.transition_id not in invalidated_ids
            ]
            memory_enabled = step >= memory_enable_after_steps
            if policy in SEQUENTIAL_POLICIES and memory_enabled:
                decision, action_image = sequential_replay_decision(
                    policy, list(memory or []), sequential_cursor, current
                )
                if action_image is not None:
                    action_source = "trajectory_replay"
                    sequential_id = str(decision["selected_transition_id"])
                    reused_actions += 1
            elif (
                policy in GUARDED_POLICIES
                and not replay_circuit_open
                and memory_enabled
            ):
                guard_select_started = time.perf_counter()
                decision, guard_context = GUARD.select(
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
                    "evaluations": [],
                    "replay_circuit_open": replay_circuit_open,
                    "fallback_reason": (
                        "memory_warm_start_prefix"
                        if policy != "always" and not memory_enabled
                        else None
                    ),
                }

            if decision["decision"] == "reuse":
                if policy in GUARDED_POLICIES:
                    if guard_context is None:
                        raise RuntimeError(
                            "reuse decision lacks authorization context"
                        )
                    action_image = guard_context.mapped_action_xy
                    action_source = "transition_guard"
                    reused_actions += 1
                elif policy not in SEQUENTIAL_POLICIES:
                    raise RuntimeError("unexpected reuse policy")
            else:
                request_id = f"{episode_id}--step-{step:02d}"
                response = grounder(request_id, before_path, goal, history)
                service_response = compact_service_response(response)
                model_calls += 1
                parsed = response["result"]["parsed_action"]
                if not parsed.get("valid"):
                    failure_reason = "base_agent_action_parse_failure"
                else:
                    action_image, boundary_clamped = normalized_to_image(
                        parsed["normalized_xy"], before
                    )
                    action_source = "base_agent"
                    decision["base_action_boundary_clamped"] = boundary_clamped

            if action_image is None:
                steps.append(
                    {
                        "step": step,
                        "before": before,
                        "history": history,
                        "post_verification": post_receipt,
                        "scoring_only_post_verification": (
                            scoring_post_receipt
                        ),
                        "guard_select_s": guard_select_s,
                        "post_verify_s": post_verify_s,
                        "scoring_only_post_verify_s": (
                            scoring_only_post_verify_s
                        ),
                        "decision": decision,
                        "service_response": service_response,
                        "action_source": action_source,
                        "action_executed": False,
                        "action_image_xy": None,
                        "action_css_xy": None,
                        "after": None,
                        "outcome": None,
                    }
                )
                break

            if not all(math.isfinite(value) for value in action_image):
                failure_reason = "nonfinite_action"
                break
            action_css = image_to_css(action_image, contract)
            _, raw_outcome = execute_action(
                env, action_css, str(action_source), step
            )
            C430.wait_for_visual_settle(env.instance.driver)
            after_path = episode_dir / f"step_{step:02d}_after{suffix}"
            after = C388.capture(env.instance.driver, after_path, contract)
            after_image = cv2.imread(str(after_path), cv2.IMREAD_COLOR)
            if after_image is None:
                raise RuntimeError(f"could not load captured image: {after_path}")
            outcome = outcome_receipt(raw_outcome)
            terminal_outcome = outcome
            history_action = {
                "action_image_xy": list(action_image),
                "source": action_source,
            }
            action_history.append(history_action)
            if action_source == "trajectory_replay":
                if sequential_id is None:
                    raise RuntimeError("sequential replay lacks transition ID")
                consumed_sequential_ids.append(sequential_id)
                sequential_cursor += 1
            elif action_source == "transition_guard":
                if guard_context is None:
                    raise RuntimeError("reused action lacks guard context")
                if policy == "full":
                    post_started = time.perf_counter()
                    post_receipt = GUARD.verify_post(guard_context, after_image)
                    post_verify_s = time.perf_counter() - post_started
                    scoring_post_receipt = post_receipt
                    if post_receipt["post_verified"]:
                        confirmed_ids.add(guard_context.transition_id)
                    else:
                        replay_circuit_open = True
                        invalidated_ids.add(guard_context.transition_id)
                else:
                    if score_postcondition_for_ablations:
                        scoring_started = time.perf_counter()
                        scoring_post_receipt = GUARD.verify_post(
                            guard_context, after_image
                        )
                        scoring_only_post_verify_s = (
                            time.perf_counter() - scoring_started
                        )
                    confirmed_ids.add(guard_context.transition_id)
            steps.append(
                {
                    "step": step,
                    "before": before,
                    "history": history,
                    "post_verification": post_receipt,
                    "scoring_only_post_verification": scoring_post_receipt,
                    "guard_select_s": guard_select_s,
                    "post_verify_s": post_verify_s,
                    "scoring_only_post_verify_s": scoring_only_post_verify_s,
                    "decision": decision,
                    "service_response": service_response,
                    "action_source": action_source,
                    "action_executed": True,
                    "action_image_xy": list(action_image),
                    "action_css_xy": action_css,
                    "after": after,
                    "outcome": outcome,
                }
            )
            if outcome["terminated"] or outcome["truncated"]:
                break
        else:
            failure_reason = "maximum_steps_exhausted"

        success = bool(
            terminal_outcome
            and terminal_outcome["terminated"]
            and float(terminal_outcome.get("raw_reward") or 0.0) >= 0.999
        )
        if not success and failure_reason is None:
            failure_reason = "environment_terminal_failure_or_incomplete"
        return {
            "runner_version": RUNNER_VERSION,
            "method_version": GUARD.METHOD_VERSION,
            "geometry_frontend": geometry_frontend,
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
            "task": task,
            "seed": seed,
            "contract": contract["name"],
            "contract_expected_global_support": bool(contract["supported"]),
            "contract_capture_scale": float(contract["capture_scale"]),
            "kind": kind,
            "policy": policy,
            "goal": goal,
            "exact_goal_memory_retrieval": memory_goal is not None,
            "memory_transitions": len(memory or []),
            "memory_enable_after_steps": memory_enable_after_steps,
            "max_steps": MAX_STEPS,
            "custom_action_executor": executor is not None,
            "score_postcondition_for_ablations": (
                score_postcondition_for_ablations
            ),
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
            "invalidated_transition_ids": sorted(invalidated_ids),
            "confirmed_transition_ids": sorted(confirmed_ids),
            "progress_recovered_transition_ids": sorted(
                progress_recovered_ids
            ),
            "consumed_sequential_transition_ids": consumed_sequential_ids,
            "replay_circuit_open": replay_circuit_open,
            "success": success,
            "failure_reason": failure_reason,
            "terminal_outcome": terminal_outcome,
            "elapsed_s": time.perf_counter() - started,
            "steps": steps,
        }
    finally:
        env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--attempt", type=int, default=0)
    parser.add_argument(
        "--contract",
        choices=(
            "source",
            "native_zoom_125",
            "native_zoom_80",
            "native_affine_skew",
            "native_jpeg_stream",
            "native_narrow_reflow_124",
        ),
        required=True,
    )
    parser.add_argument(
        "--policy",
        choices=("always", "raw", "naive", "sg", "sgp", "full"),
        required=True,
    )
    parser.add_argument("--kind", choices=("qwen25vl", "uitars"), required=True)
    parser.add_argument("--service-url", required=True)
    parser.add_argument(
        "--geometry-frontend",
        choices=("lightglue", "sift"),
        default="lightglue",
    )
    parser.add_argument("--memory-episode", type=Path)
    parser.add_argument("--risk-certificate", type=Path)
    parser.add_argument("--require-risk-certificate", action="store_true")
    parser.add_argument("--memory-enable-after-steps", type=int, default=0)
    parser.add_argument(
        "--score-postcondition-for-ablations", action="store_true"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.out = args.out.resolve()
    if args.memory_episode is not None:
        args.memory_episode = args.memory_episode.resolve()
    if args.risk_certificate is not None:
        args.risk_certificate = args.risk_certificate.resolve()
    if args.require_risk_certificate and args.policy != "full":
        raise ValueError("--require-risk-certificate is valid only for full")
    if args.require_risk_certificate and args.risk_certificate is None:
        raise ValueError("required risk certificate path is missing")
    if args.risk_certificate is not None and args.policy != "full":
        raise ValueError("risk certificate is valid only for full policy")
    if args.policy == "always":
        memory_goal, memory = None, None
    else:
        if args.memory_episode is None:
            raise ValueError("replay policy requires --memory-episode")
        memory_goal, memory = load_memory(args.memory_episode)
    risk_certificate = (
        load_risk_certificate(args.risk_certificate)
        if args.risk_certificate is not None
        else None
    )

    service_health = CLIENT.health(args.service_url)
    if not service_health.get("ready") or service_health.get("kind") != args.kind:
        raise RuntimeError("model service health/kind mismatch")

    def grounder(
        request_id: str, image_path: Path, goal: str, history: str
    ) -> dict[str, Any]:
        return service_grounder(
            args.service_url,
            args.kind,
            request_id,
            image_path,
            goal,
            history,
        )

    frontend = (
        LIGHTGLUE.LightGlueRuntimeFrontend()
        if args.policy in GUARDED_POLICIES
        and args.geometry_frontend == "lightglue"
        else None
    )
    result = run_episode(
        task=args.task,
        seed=args.seed,
        contract=contract_by_name(args.contract),
        policy=args.policy,
        kind=args.kind,
        output_dir=args.output_dir,
        grounder=grounder,
        run_id=args.run_id,
        attempt=args.attempt,
        memory_goal=memory_goal,
        memory=memory,
        match_provider=(frontend.correspondences if frontend is not None else None),
        geometry_frontend=(
            frontend.bindings["frontend_version"]
            if frontend is not None
            else (
                "not_applicable_sequential_replay"
                if args.policy in SEQUENTIAL_POLICIES
                else "classical_sift"
            )
        ),
        risk_certificate=risk_certificate,
        require_risk_certificate=args.require_risk_certificate,
        memory_enable_after_steps=args.memory_enable_after_steps,
        score_postcondition_for_ablations=(
            args.score_postcondition_for_ablations
        ),
    )
    result["services"] = {"model": service_health}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.out)
    print(
        json.dumps(
            {
                "episode_id": result["episode_id"],
                "success": result["success"],
                "executed_actions": result["executed_actions"],
                "model_calls": result["model_calls"],
                "reused_actions": result["reused_actions"],
                "failure_reason": result["failure_reason"],
            },
            indent=2,
        )
    )
    return 0 if result["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
