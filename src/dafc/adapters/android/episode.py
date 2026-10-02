from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

import dafc.transitions as GUARD
from dafc.adapters.android.device import AndroidDevice
from dafc.adapters.android.episode_report import summarize_episode
from dafc.adapters.android.episode_state import EpisodeConfig, EpisodeState, StepState
from dafc.adapters.android.episode_steps import (
    confirm_replay,
    create_episode,
    execute_step,
    initialize_episode,
    observe_step,
    record_executed,
    record_unexecuted,
    resolve_action,
    select_memory,
    validate_identity,
    validate_replay,
)
from dafc.adapters.android.observations import MAX_STEPS


def run_episode(
    *,
    env: Any,
    task_registry: dict[str, Any],
    task_id: str,
    contract_id: str,
    policy: str,
    kind: str,
    model_service_url: str,
    run_id: str,
    attempt: int,
    output_root: Path,
    timer_params: dict[str, Any] | None = None,
    browser_task_seed: Any | None = None,
    memory_goal: str | None = None,
    memory_params: dict[str, Any] | None = None,
    memory: list[GUARD.Transition] | None = None,
    match_provider: Callable[[np.ndarray, np.ndarray], dict[str, Any]] | None = None,
    risk_certificate: GUARD.RiskCertificate | None = None,
    require_risk_certificate: bool = False,
    score_postcondition_for_ablations: bool = False,
    device: AndroidDevice,
) -> dict[str, Any]:
    config = EpisodeConfig(
        env=env,
        task_registry=task_registry,
        task_id=task_id,
        contract_id=contract_id,
        policy=policy,
        kind=kind,
        model_service_url=model_service_url,
        run_id=run_id,
        attempt=attempt,
        output_root=output_root,
        timer_params=timer_params,
        browser_task_seed=browser_task_seed,
        memory_goal=memory_goal,
        memory_params=memory_params,
        memory=memory,
        match_provider=match_provider,
        risk_certificate=risk_certificate,
        require_risk_certificate=require_risk_certificate,
        score_postcondition_for_ablations=score_postcondition_for_ablations,
        device=device,
    )
    validate_identity(config)
    validate_replay(config)
    state = EpisodeState()
    create_episode(config, state)
    try:
        initialize_episode(config, state)
        for index in range(MAX_STEPS):
            frame = StepState(index=index)
            observe_step(config, state, frame)
            select_memory(config, state, frame)
            resolve_action(config, state, frame)
            if frame.action_xy is None:
                record_unexecuted(config, state, frame)
                break
            execute_step(config, state, frame)
            confirm_replay(config, state, frame)
            record_executed(config, state, frame)
            if state.final_reward == 1.0:
                break
        else:
            state.failure_reason = "maximum_steps_exhausted"
        return summarize_episode(config, state)
    finally:
        state.task.tear_down(config.env)
