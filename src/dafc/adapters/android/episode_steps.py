from __future__ import annotations

import time

import dafc.adapters.android.device as C482
import dafc.models.client as GROUND_CLIENT
import dafc.transitions as GUARD
from dafc.adapters.android.episode_state import EpisodeConfig, EpisodeState, StepState
from dafc.adapters.android.observations import (
    CONTRACTS,
    GUARDED_POLICIES,
    SEQUENTIAL_POLICIES,
    capture,
    compact_model_response,
    ensure_clean_observation_ui,
    execute_click,
    history_text,
    instruction,
    normalized_to_pixel,
    open_app,
    parameter_fingerprint,
    resolved_task_spec,
    sequential_replay_decision,
)


def validate_identity(config: EpisodeConfig):
    """Validate task identity, actor and output names before platform mutation."""
    if config.contract_id not in CONTRACTS:
        raise ValueError(f"unknown Android contract: {config.contract_id}")
    if config.policy not in {"always", *SEQUENTIAL_POLICIES, *GUARDED_POLICIES}:
        raise ValueError(f"unknown policy: {config.policy}")
    if config.kind not in {"qwen25vl", "uitars"}:
        raise ValueError(f"unknown model kind: {config.kind}")
    if config.attempt < 0:
        raise ValueError("attempt must be non-negative")
    if (
        not config.run_id
        or len(config.run_id) > 80
        or any((not (ch.isalnum() or ch in "-_.") for ch in config.run_id))
    ):
        raise ValueError("run_id must be a 1-80 character safe path component")


def validate_replay(config: EpisodeConfig):
    """Require successful memory and matching providers for guarded replay."""
    if config.policy != "always" and (
        config.memory_goal is None or config.memory_params is None or config.memory is None
    ):
        raise ValueError("replay policies require successful memory")
    if config.policy in GUARDED_POLICIES and config.match_provider is None:
        raise ValueError("guarded replay policies require a match provider")
    if config.require_risk_certificate and config.policy != "full":
        raise ValueError("risk-limited activation applies only to full policy")
    if config.score_postcondition_for_ablations and config.policy not in GUARDED_POLICIES:
        raise ValueError("scoring-only postconditions require a guarded replay policy")


def create_episode(config: EpisodeConfig, state: EpisodeState):
    """Apply the fixed display contract and construct the task before initialization."""
    state.spec = resolved_task_spec(config.task_id, config.timer_params, config.browser_task_seed)
    state.contract = CONTRACTS[config.contract_id]
    state.params_fingerprint = parameter_fingerprint(state.spec.params)
    state.episode_id = f"{config.run_id}--{config.task_id}--p{state.params_fingerprint[:12]}--{config.contract_id}--{config.kind}--{config.policy}--a{config.attempt:02d}"
    state.episode_dir = config.output_root / state.episode_id
    config.device.apply_contract(state.contract)
    config.env.reset(go_home=True)
    if config.task_id == "AudioRecorderRecordAudio":
        from android_world.env.setup_device import apps as setup_apps

        setup_apps.AudioRecorder.setup(config.env)
    state.task = config.task_registry[config.task_id](dict(state.spec.params))
    state.started = time.perf_counter()
    state.steps = []
    state.history_actions = []
    state.model_calls = 0
    state.reused_actions = 0
    state.confirmed_ids = set()
    state.progress_recovered_ids = set()
    state.invalidated_ids = set()
    state.consumed_sequential_ids = []
    state.sequential_cursor = 0
    state.replay_circuit_open = False
    state.failure_reason = None
    state.goal = None
    state.initial_reward = None
    state.final_reward = 0.0
    state.observation_ui_receipts = []


def initialize_episode(config: EpisodeConfig, state: EpisodeState):
    """Initialize the task and validate exact goal and parameter retrieval."""
    state.task.initialize_task(config.env)
    state.observation_ui_receipts.append(
        ensure_clean_observation_ui(config.env, "after_task_initialization", device=config.device)
    )
    state.initial_reward = float(state.task.is_successful(config.env))
    open_app(config.env, state.spec.app_name)
    state.observation_ui_receipts.append(
        ensure_clean_observation_ui(config.env, "after_open_app", device=config.device)
    )
    raw_state = config.env.get_state(wait_to_stabilize=True)
    state.goal = str(state.task.goal)
    if config.memory_goal is not None and state.goal != config.memory_goal:
        raise RuntimeError("Android exact-goal memory retrieval mismatch")
    if config.memory_params is not None and dict(state.spec.params) != dict(config.memory_params):
        raise RuntimeError("Android exact-parameter memory retrieval mismatch")
    current_raw = C482.pixels(raw_state)
    state.current = C482.observe(current_raw, state.contract)


def observe_step(config: EpisodeConfig, state: EpisodeState, frame: StepState):
    """Capture the current Android observation for decision making."""
    frame.before_path = state.episode_dir / f"step_{frame.index:02d}_before.png"
    frame.before_receipt = capture(frame.before_path, state.current)
    frame.history = history_text(state.history_actions)
    frame.context = None
    frame.model_response = None
    frame.action_xy = None
    frame.action_source = None
    frame.sequential_id = None
    frame.guard_select_s = 0.0
    frame.post_verify_s = 0.0
    frame.scoring_only_post_verify_s = 0.0


def select_memory(config: EpisodeConfig, state: EpisodeState, frame: StepState):
    """Authorize reuse using source evidence and current pixels only."""
    available_memory = [row for row in config.memory or [] if row.transition_id not in state.invalidated_ids]
    if config.policy in SEQUENTIAL_POLICIES:
        frame.decision, frame.action_xy = sequential_replay_decision(
            config.policy, list(config.memory or []), state.sequential_cursor, state.current
        )
        if frame.action_xy is not None:
            frame.action_source = "trajectory_replay"
            frame.sequential_id = str(frame.decision["selected_transition_id"])
            state.reused_actions += 1
    elif config.policy in GUARDED_POLICIES and (not state.replay_circuit_open):
        guard_select_started = time.perf_counter()
        frame.decision, frame.context = GUARD.select(
            available_memory,
            state.current,
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
            "confirmed_transition_ids": sorted(state.confirmed_ids),
            "evaluations": [],
            "replay_circuit_open": state.replay_circuit_open,
        }


def resolve_action(config: EpisodeConfig, state: EpisodeState, frame: StepState):
    """Select the authorized point or one model-generated fallback."""
    if frame.decision["decision"] == "reuse":
        if config.policy in GUARDED_POLICIES:
            if frame.context is None:
                raise RuntimeError("reuse decision lacks authorization context")
            frame.action_xy = frame.context.mapped_action_xy
            frame.action_source = "transition_guard"
            state.reused_actions += 1
        elif config.policy not in SEQUENTIAL_POLICIES:
            raise RuntimeError("unexpected reuse policy")
    else:
        request_id = f"{state.episode_id}--step-{frame.index:02d}"
        response = GROUND_CLIENT.infer(
            config.model_service_url,
            request_id,
            frame.before_path,
            instruction=instruction(state.goal, frame.history),
        )
        frame.model_response = compact_model_response(response)
        state.model_calls += 1
        parsed = response["result"]["parsed_action"]
        if not parsed.get("valid"):
            state.failure_reason = "base_agent_action_parse_failure"
        else:
            frame.action_xy, clamped = normalized_to_pixel(parsed["normalized_xy"], frame.before_receipt)
            frame.action_source = "base_agent"
            frame.decision["base_action_boundary_clamped"] = clamped


def record_unexecuted(config: EpisodeConfig, state: EpisodeState, frame: StepState):
    """Preserve a parse failure without executing an action."""
    state.steps.append(
        {
            "step": frame.index,
            "before": frame.before_receipt,
            "history": frame.history,
            "decision": frame.decision,
            "model_response": frame.model_response,
            "action_source": frame.action_source,
            "action_executed": False,
            "action_xy": None,
            "after": None,
            "post_verification": None,
            "scoring_only_post_verification": None,
            "guard_select_s": frame.guard_select_s,
            "post_verify_s": frame.post_verify_s,
            "scoring_only_post_verify_s": frame.scoring_only_post_verify_s,
            "official_reward_after_action": None,
        }
    )


def execute_step(config: EpisodeConfig, state: EpisodeState, frame: StepState):
    """Execute the bounded point and capture the next observation."""
    requested_action_xy = frame.action_xy
    frame.action_xy = execute_click(config.env, requested_action_xy, state.current)
    frame.decision["requested_action_xy"] = list(requested_action_xy)
    frame.decision["executed_action_xy"] = list(frame.action_xy)
    state.observation_ui_receipts.append(
        ensure_clean_observation_ui(config.env, f"after_step_{frame.index:02d}_action", device=config.device)
    )
    raw_state = config.env.get_state(wait_to_stabilize=True)
    next_raw = C482.pixels(raw_state)
    frame.next_observation = C482.observe(next_raw, state.contract)
    after_path = state.episode_dir / f"step_{frame.index:02d}_after.png"
    frame.after_receipt = capture(after_path, frame.next_observation)
    frame.post = None
    frame.scoring_post = None


def confirm_replay(config: EpisodeConfig, state: EpisodeState, frame: StepState):
    """Confirm replay progress or disable further replay after a failed postcondition."""
    if frame.action_source == "trajectory_replay":
        if frame.sequential_id is None:
            raise RuntimeError("sequential replay lacks transition ID")
        state.consumed_sequential_ids.append(frame.sequential_id)
        state.sequential_cursor += 1
    elif frame.action_source == "transition_guard":
        if config.policy == "full":
            post_started = time.perf_counter()
            frame.post = GUARD.verify_post(frame.context, frame.next_observation)
            frame.post_verify_s = time.perf_counter() - post_started
            frame.scoring_post = frame.post
        elif config.score_postcondition_for_ablations:
            scoring_started = time.perf_counter()
            frame.scoring_post = GUARD.verify_post(frame.context, frame.next_observation)
            frame.scoring_only_post_verify_s = time.perf_counter() - scoring_started
        if frame.post is None or frame.post["post_verified"]:
            state.confirmed_ids.add(frame.context.transition_id)
        else:
            state.invalidated_ids.add(frame.context.transition_id)
            state.replay_circuit_open = True


def record_executed(config: EpisodeConfig, state: EpisodeState, frame: StepState):
    """Score only after execution and preserve the full step record."""
    state.final_reward = float(state.task.is_successful(config.env))
    state.steps.append(
        {
            "step": frame.index,
            "before": frame.before_receipt,
            "history": frame.history,
            "decision": frame.decision,
            "model_response": frame.model_response,
            "action_source": frame.action_source,
            "action_executed": True,
            "action_xy": list(frame.action_xy),
            "after": frame.after_receipt,
            "post_verification": frame.post,
            "scoring_only_post_verification": frame.scoring_post,
            "guard_select_s": frame.guard_select_s,
            "post_verify_s": frame.post_verify_s,
            "scoring_only_post_verify_s": frame.scoring_only_post_verify_s,
            "official_reward_after_action": state.final_reward,
        }
    )
    state.history_actions.append({"action_xy": list(frame.action_xy), "source": frame.action_source})
    state.current = frame.next_observation
