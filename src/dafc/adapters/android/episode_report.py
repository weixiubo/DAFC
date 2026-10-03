from __future__ import annotations

import time

import dafc.transitions as GUARD
from dafc.adapters.android.episode_state import EpisodeConfig, EpisodeState
from dafc.adapters.android.observations import (
    MAX_STEPS,
    OPEN_APP_SETTLE_S,
    RUNNER_VERSION,
)


def summarize_episode(config: EpisodeConfig, state: EpisodeState):
    """Summarize rewards, operational decisions and scoring-only diagnostics."""
    success = state.final_reward == 1.0
    if not success and state.failure_reason is None:
        state.failure_reason = "official_task_incomplete"
    return {
        "runner_version": RUNNER_VERSION,
        "method_version": GUARD.METHOD_VERSION,
        "risk_certificate_required": config.require_risk_certificate,
        "risk_certificate": config.risk_certificate.receipt()
        if config.risk_certificate is not None
        else None,
        "paper_evidence": False,
        "episode_id": state.episode_id,
        "run_id": config.run_id,
        "attempt": config.attempt,
        "task_id": config.task_id,
        "params": state.spec.params,
        "parameter_fingerprint": state.params_fingerprint,
        "goal": state.goal,
        "contract": config.contract_id,
        "policy": config.policy,
        "kind": config.kind,
        "nonpoint_open_app_bootstrap": state.spec.app_name,
        "open_app_settle_s": OPEN_APP_SETTLE_S,
        "observation_ui": {
            "required_pointer_location": "0",
            "all_receipts_clean": all(
                (row["pointer_location_after_cleanup"] == "0" for row in state.observation_ui_receipts)
            ),
            "receipts": state.observation_ui_receipts,
        },
        "max_steps": MAX_STEPS,
        "initial_reward": state.initial_reward,
        "final_reward": state.final_reward,
        "success": success,
        "failure_reason": state.failure_reason,
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
        "score_postcondition_for_ablations": config.score_postcondition_for_ablations,
        "confirmed_transition_ids": sorted(state.confirmed_ids),
        "progress_recovered_transition_ids": sorted(state.progress_recovered_ids),
        "consumed_sequential_transition_ids": state.consumed_sequential_ids,
        "invalidated_transition_ids": sorted(state.invalidated_ids),
        "replay_circuit_open": state.replay_circuit_open,
        "elapsed_s": time.perf_counter() - state.started,
        "steps": state.steps,
        "input_boundary": {
            "model": ["current screenshot", "goal", "pixel action history"],
            "guard": ["successful source transition pixels/actions", "current and next target screenshots"],
            "excluded": [
                "target accessibility tree",
                "target element labels",
                "target action oracle",
                "official reward before post-action scoring",
            ],
        },
    }
