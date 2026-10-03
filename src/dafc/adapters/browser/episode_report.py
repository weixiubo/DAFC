from __future__ import annotations

import time

import dafc.transitions as GUARD
from dafc.adapters.browser.episode_state import EpisodeConfig, EpisodeState
from dafc.adapters.browser.observations import (
    MAX_STEPS,
    RUNNER_VERSION,
)


def summarize_episode(config: EpisodeConfig, state: EpisodeState):
    """Summarize the completed episode without altering step evidence."""
    success = bool(
        state.terminal_outcome
        and state.terminal_outcome["terminated"]
        and (float(state.terminal_outcome.get("raw_reward") or 0.0) >= 0.999)
    )
    if not success and state.failure_reason is None:
        state.failure_reason = "environment_terminal_failure_or_incomplete"
    return {
        "runner_version": RUNNER_VERSION,
        "method_version": GUARD.METHOD_VERSION,
        "geometry_frontend": config.geometry_frontend,
        "risk_certificate_required": config.require_risk_certificate,
        "risk_certificate": config.risk_certificate.receipt()
        if config.risk_certificate is not None
        else None,
        "paper_evidence": False,
        "episode_id": state.episode_id,
        "run_id": config.run_id,
        "attempt": config.attempt,
        "task": config.task,
        "seed": config.seed,
        "contract": config.contract["name"],
        "contract_expected_global_support": bool(config.contract["supported"]),
        "contract_capture_scale": float(config.contract["capture_scale"]),
        "kind": config.kind,
        "policy": config.policy,
        "goal": state.goal,
        "exact_goal_memory_retrieval": config.memory_goal is not None,
        "memory_transitions": len(config.memory or []),
        "memory_enable_after_steps": config.memory_enable_after_steps,
        "max_steps": MAX_STEPS,
        "custom_action_executor": config.executor is not None,
        "score_postcondition_for_ablations": config.score_postcondition_for_ablations,
        "executed_actions": sum((row["action_executed"] for row in state.steps)),
        "model_calls": state.model_calls,
        "reused_actions": state.reused_actions,
        "postcondition_failures": sum(
            (
                bool(row["post_verification"] and (not row["post_verification"]["post_verified"]))
                for row in state.steps
            )
        ),
        "scoring_only_postcondition_failures": sum(
            (
                bool(
                    row.get("scoring_only_post_verification")
                    and (not row["scoring_only_post_verification"]["post_verified"])
                )
                for row in state.steps
            )
        ),
        "operational_guard_s": sum(
            (
                float(row.get("guard_select_s", 0.0)) + float(row.get("post_verify_s", 0.0))
                for row in state.steps
            )
        ),
        "scoring_only_audit_s": sum(
            (float(row.get("scoring_only_post_verify_s", 0.0)) for row in state.steps)
        ),
        "invalidated_transition_ids": sorted(state.invalidated_ids),
        "confirmed_transition_ids": sorted(state.confirmed_ids),
        "progress_recovered_transition_ids": sorted(state.progress_recovered_ids),
        "consumed_sequential_transition_ids": state.consumed_sequential_ids,
        "replay_circuit_open": state.replay_circuit_open,
        "success": success,
        "failure_reason": state.failure_reason,
        "terminal_outcome": state.terminal_outcome,
        "elapsed_s": time.perf_counter() - state.started,
        "steps": state.steps,
    }
