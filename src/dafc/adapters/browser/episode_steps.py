from __future__ import annotations

import time

import cv2

import dafc.adapters.browser.capture as C388
import dafc.adapters.browser.execution as C400
import dafc.adapters.browser.interactions as C430
import dafc.adapters.browser.miniwob_agent_harness as HARNESS
import dafc.transitions as GUARD
from dafc.adapters.browser.episode_state import EpisodeConfig, EpisodeState, StepState
from dafc.adapters.browser.observations import (
    GUARDED_POLICIES,
    SEQUENTIAL_POLICIES,
    TASKS,
    compact_service_response,
    history_text,
    image_to_css,
    normalized_to_image,
    outcome_receipt,
    sequential_replay_decision,
)


def validate_identity(config: EpisodeConfig):
    """Validate cohort and safe run identifiers before constructing the browser."""
    if config.task not in TASKS:
        raise ValueError(f"task is outside the frozen eight-task cohort: {config.task}")
    if config.policy not in {"always", *SEQUENTIAL_POLICIES, *GUARDED_POLICIES}:
        raise ValueError(f"unknown policy: {config.policy}")
    if config.attempt < 0:
        raise ValueError("attempt must be non-negative")
    if (
        not config.run_id
        or len(config.run_id) > 80
        or any((not (ch.isalnum() or ch in "-_.") for ch in config.run_id))
    ):
        raise ValueError("run_id must be a 1-80 character safe path component")


def validate_replay(config: EpisodeConfig):
    """Validate replay policy combinations independently of browser setup."""
    if config.policy != "always" and (config.memory_goal is None or config.memory is None):
        raise ValueError("replay policies require a successful source memory")
    if config.require_risk_certificate and config.policy != "full":
        raise ValueError("risk-limited activation applies only to full policy")
    if config.score_postcondition_for_ablations and config.policy not in GUARDED_POLICIES:
        raise ValueError("scoring-only postconditions require a guarded replay policy")
    if config.memory_enable_after_steps < 0:
        raise ValueError("memory_enable_after_steps must be non-negative")
    if config.policy == "always" and config.memory_enable_after_steps != 0:
        raise ValueError("always policy cannot have a memory warm-start prefix")


def initialize_episode(config: EpisodeConfig, state: EpisodeState):
    """Reset the target once and initialize the episode state."""
    state.episode_id = f"{config.run_id}--{config.task}--{config.seed}--{config.contract['name']}--{config.kind}--{config.policy}--a{config.attempt:02d}"
    state.episode_dir = config.output_dir / state.episode_id
    state.started = time.perf_counter()
    state.env.instance.driver.set_window_size(900, 1000)
    observation, _ = HARNESS.reset_agent_episode(state.env, config.seed)
    C400.reset_harness(state.env.instance.driver)
    state.goal = str(observation["utterance"])
    if config.memory_goal is not None and state.goal != config.memory_goal:
        raise RuntimeError("exact-goal memory retrieval mismatch")
    if config.contract.get("script"):
        state.env.instance.driver.execute_script(config.contract["script"])
    state.suffix = ".jpg" if config.contract["format"] == "jpeg" else ".png"
    state.action_history = []
    state.steps = []
    state.model_calls = 0
    state.reused_actions = 0
    state.invalidated_ids = set()
    state.confirmed_ids = set()
    state.progress_recovered_ids = set()
    state.consumed_sequential_ids = []
    state.sequential_cursor = 0
    state.replay_circuit_open = False
    state.terminal_outcome = None
    state.failure_reason = None
    state.execute_action = config.executor or (
        lambda active_env, point, _source, _step: C430.click(active_env, point)
    )


def observe_step(config: EpisodeConfig, state: EpisodeState, frame: StepState):
    """Capture the only screenshot available to the action policy."""
    frame.before_path = state.episode_dir / f"step_{frame.index:02d}_before{state.suffix}"
    frame.before = C388.capture(state.env.instance.driver, frame.before_path, config.contract)
    frame.current = cv2.imread(str(frame.before_path), cv2.IMREAD_COLOR)
    if frame.current is None:
        raise RuntimeError(f"could not load captured image: {frame.before_path}")
    frame.post_receipt = None
    frame.scoring_post_receipt = None
    frame.guard_select_s = 0.0
    frame.post_verify_s = 0.0
    frame.scoring_only_post_verify_s = 0.0
    frame.history = history_text(state.action_history)
    frame.service_response = None
    frame.action_image = None
    frame.action_source = None
    frame.guard_context = None
    frame.sequential_id = None


def select_memory(config: EpisodeConfig, state: EpisodeState, frame: StepState):
    """Authorize reuse and track confirmed progress without reading target reward."""
    available_memory = [row for row in config.memory or [] if row.transition_id not in state.invalidated_ids]
    memory_enabled = frame.index >= config.memory_enable_after_steps
    if config.policy in SEQUENTIAL_POLICIES and memory_enabled:
        frame.decision, frame.action_image = sequential_replay_decision(
            config.policy, list(config.memory or []), state.sequential_cursor, frame.current
        )
        if frame.action_image is not None:
            frame.action_source = "trajectory_replay"
            frame.sequential_id = str(frame.decision["selected_transition_id"])
            state.reused_actions += 1
    elif config.policy in GUARDED_POLICIES and (not state.replay_circuit_open) and memory_enabled:
        guard_select_started = time.perf_counter()
        frame.decision, frame.guard_context = GUARD.select(
            available_memory,
            frame.current,
            require_precondition=config.policy != "sg",
            confirmed_transition_ids=state.confirmed_ids,
            match_provider=config.match_provider,
            risk_certificate=config.risk_certificate,
            require_risk_certificate=config.require_risk_certificate,
        )
        frame.guard_select_s = time.perf_counter() - guard_select_started
        recovered_now = set(frame.decision.get("progress_recovered_transition_ids", []))
        known_memory_ids = {row.transition_id for row in available_memory}
        if not recovered_now.issubset(known_memory_ids):
            raise RuntimeError("guard recovered progress outside available memory")
        state.confirmed_ids.update(recovered_now)
        state.progress_recovered_ids.update(recovered_now)
    else:
        frame.decision = {
            "method_version": GUARD.METHOD_VERSION,
            "decision": "fallback",
            "selected_transition_id": None,
            "selected_demonstration_order": None,
            "mapped_action_xy": None,
            "evaluations": [],
            "replay_circuit_open": state.replay_circuit_open,
            "fallback_reason": "memory_warm_start_prefix"
            if config.policy != "always" and (not memory_enabled)
            else None,
        }


def resolve_action(config: EpisodeConfig, state: EpisodeState, frame: StepState):
    """Use an authorized action or ask the same target actor once."""
    if frame.decision["decision"] == "reuse":
        if config.policy in GUARDED_POLICIES:
            if frame.guard_context is None:
                raise RuntimeError("reuse decision lacks authorization context")
            frame.action_image = frame.guard_context.mapped_action_xy
            frame.action_source = "transition_guard"
            state.reused_actions += 1
        elif config.policy not in SEQUENTIAL_POLICIES:
            raise RuntimeError("unexpected reuse policy")
    else:
        request_id = f"{state.episode_id}--step-{frame.index:02d}"
        response = config.grounder(request_id, frame.before_path, state.goal, frame.history)
        frame.service_response = compact_service_response(response)
        state.model_calls += 1
        parsed = response["result"]["parsed_action"]
        if not parsed.get("valid"):
            state.failure_reason = "base_agent_action_parse_failure"
        else:
            frame.action_image, boundary_clamped = normalized_to_image(parsed["normalized_xy"], frame.before)
            frame.action_source = "base_agent"
            frame.decision["base_action_boundary_clamped"] = boundary_clamped


def record_unexecuted(config: EpisodeConfig, state: EpisodeState, frame: StepState):
    """Retain a failed parse as an outcome-bearing step."""
    state.steps.append(
        {
            "step": frame.index,
            "before": frame.before,
            "history": frame.history,
            "post_verification": frame.post_receipt,
            "scoring_only_post_verification": frame.scoring_post_receipt,
            "guard_select_s": frame.guard_select_s,
            "post_verify_s": frame.post_verify_s,
            "scoring_only_post_verify_s": frame.scoring_only_post_verify_s,
            "decision": frame.decision,
            "service_response": frame.service_response,
            "action_source": frame.action_source,
            "action_executed": False,
            "action_image_xy": None,
            "action_css_xy": None,
            "after": None,
            "outcome": None,
        }
    )


def execute_step(config: EpisodeConfig, state: EpisodeState, frame: StepState):
    """Execute in CSS coordinates, capture the result and record target outcome."""
    frame.action_css = image_to_css(frame.action_image, config.contract)
    _, raw_outcome = state.execute_action(state.env, frame.action_css, str(frame.action_source), frame.index)
    C430.wait_for_visual_settle(state.env.instance.driver)
    after_path = state.episode_dir / f"step_{frame.index:02d}_after{state.suffix}"
    frame.after = C388.capture(state.env.instance.driver, after_path, config.contract)
    frame.after_image = cv2.imread(str(after_path), cv2.IMREAD_COLOR)
    if frame.after_image is None:
        raise RuntimeError(f"could not load captured image: {after_path}")
    frame.outcome = outcome_receipt(raw_outcome)
    state.terminal_outcome = frame.outcome
    history_action = {"action_image_xy": list(frame.action_image), "source": frame.action_source}
    state.action_history.append(history_action)


def confirm_replay(config: EpisodeConfig, state: EpisodeState, frame: StepState):
    """Apply postconditions and close the replay circuit on operational failure."""
    if frame.action_source == "trajectory_replay":
        if frame.sequential_id is None:
            raise RuntimeError("sequential replay lacks transition ID")
        state.consumed_sequential_ids.append(frame.sequential_id)
        state.sequential_cursor += 1
    elif frame.action_source == "transition_guard":
        if frame.guard_context is None:
            raise RuntimeError("reused action lacks guard context")
        if config.policy == "full":
            post_started = time.perf_counter()
            frame.post_receipt = GUARD.verify_post(frame.guard_context, frame.after_image)
            frame.post_verify_s = time.perf_counter() - post_started
            frame.scoring_post_receipt = frame.post_receipt
            if frame.post_receipt["post_verified"]:
                state.confirmed_ids.add(frame.guard_context.transition_id)
            else:
                state.replay_circuit_open = True
                state.invalidated_ids.add(frame.guard_context.transition_id)
        else:
            if config.score_postcondition_for_ablations:
                scoring_started = time.perf_counter()
                frame.scoring_post_receipt = GUARD.verify_post(frame.guard_context, frame.after_image)
                frame.scoring_only_post_verify_s = time.perf_counter() - scoring_started
            state.confirmed_ids.add(frame.guard_context.transition_id)


def record_executed(config: EpisodeConfig, state: EpisodeState, frame: StepState):
    """Record action, outcome and postcondition evidence together."""
    state.steps.append(
        {
            "step": frame.index,
            "before": frame.before,
            "history": frame.history,
            "post_verification": frame.post_receipt,
            "scoring_only_post_verification": frame.scoring_post_receipt,
            "guard_select_s": frame.guard_select_s,
            "post_verify_s": frame.post_verify_s,
            "scoring_only_post_verify_s": frame.scoring_only_post_verify_s,
            "decision": frame.decision,
            "service_response": frame.service_response,
            "action_source": frame.action_source,
            "action_executed": True,
            "action_image_xy": list(frame.action_image),
            "action_css_xy": frame.action_css,
            "after": frame.after,
            "outcome": frame.outcome,
        }
    )
