from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miniwob.environment import MiniWoBEnvironment

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from dafc.adapters.browser.episode_report import summarize_episode
from dafc.adapters.browser.episode_state import EpisodeConfig, EpisodeState, StepState
from dafc.adapters.browser.episode_steps import (
    confirm_replay,
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
from dafc.adapters.browser.observations import (
    MAX_STEPS,
)


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
        [MiniWoBEnvironment, list[float], str, int], tuple[dict[str, Any] | None, dict[str, Any]]
    ]
    | None = None,
    risk_certificate: Any | None = None,
    require_risk_certificate: bool = False,
    memory_enable_after_steps: int = 0,
    score_postcondition_for_ablations: bool = False,
    environment_factory: Any = None,
) -> dict[str, Any]:
    config = EpisodeConfig(
        task=task,
        seed=seed,
        contract=contract,
        policy=policy,
        kind=kind,
        output_dir=output_dir,
        grounder=grounder,
        run_id=run_id,
        attempt=attempt,
        memory_goal=memory_goal,
        memory=memory,
        match_provider=match_provider,
        geometry_frontend=geometry_frontend,
        executor=executor,
        risk_certificate=risk_certificate,
        require_risk_certificate=require_risk_certificate,
        memory_enable_after_steps=memory_enable_after_steps,
        score_postcondition_for_ablations=score_postcondition_for_ablations,
    )
    validate_identity(config)
    validate_replay(config)
    if environment_factory is None:
        from miniwob.environment import MiniWoBEnvironment

        environment_factory = MiniWoBEnvironment
    state = EpisodeState(env=environment_factory(subdomain=config.task, action_space_config="all_supported"))
    try:
        initialize_episode(config, state)
        for index in range(MAX_STEPS):
            frame = StepState(index=index)
            observe_step(config, state, frame)
            select_memory(config, state, frame)
            resolve_action(config, state, frame)
            if frame.action_image is None:
                record_unexecuted(config, state, frame)
                break
            if not all((math.isfinite(value) for value in frame.action_image)):
                state.failure_reason = "nonfinite_action"
                break
            execute_step(config, state, frame)
            confirm_replay(config, state, frame)
            record_executed(config, state, frame)
            if frame.outcome["terminated"] or frame.outcome["truncated"]:
                break
        else:
            state.failure_reason = "maximum_steps_exhausted"
        return summarize_episode(config, state)
    finally:
        state.env.close()
