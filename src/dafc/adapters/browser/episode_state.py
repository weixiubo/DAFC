from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from miniwob.environment import MiniWoBEnvironment

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True, kw_only=True)
class EpisodeConfig:
    task: str
    seed: int
    contract: dict[str, Any]
    policy: str
    kind: str
    output_dir: Path
    grounder: Callable[[str, Path, str, str], dict[str, Any]]
    run_id: str = "development-legacy"
    attempt: int = 0
    memory_goal: str | None = None
    memory: list[Any] | None = None
    match_provider: Callable[[np.ndarray, np.ndarray], dict[str, Any]] | None = None
    geometry_frontend: str = "classical_sift"
    executor: (
        Callable[[MiniWoBEnvironment, list[float], str, int], tuple[dict[str, Any] | None, dict[str, Any]]]
        | None
    ) = None
    risk_certificate: Any | None = None
    require_risk_certificate: bool = False
    memory_enable_after_steps: int = 0
    score_postcondition_for_ablations: bool = False


@dataclass
class EpisodeState:
    episode_id: str = ""
    episode_dir: Path = Path()
    started: float = 0.0
    goal: str | None = None
    steps: list[dict[str, Any]] = field(default_factory=list)
    model_calls: int = 0
    reused_actions: int = 0
    sequential_cursor: int = 0
    invalidated_ids: set[str] = field(default_factory=set)
    confirmed_ids: set[str] = field(default_factory=set)
    progress_recovered_ids: set[str] = field(default_factory=set)
    consumed_sequential_ids: list[str] = field(default_factory=list)
    replay_circuit_open: bool = False
    failure_reason: str | None = None
    env: Any = None
    suffix: str = ".png"
    action_history: list[dict[str, Any]] = field(default_factory=list)
    terminal_outcome: dict[str, Any] | None = None
    execute_action: Any = None


@dataclass
class StepState:
    index: int = 0
    before_path: Path = Path()
    history: str = ""
    decision: dict[str, Any] = field(default_factory=dict)
    action_source: str | None = None
    sequential_id: str | None = None
    guard_select_s: float = 0.0
    post_verify_s: float = 0.0
    scoring_only_post_verify_s: float = 0.0
    before: dict[str, Any] = field(default_factory=dict)
    current: np.ndarray | None = None
    post_receipt: dict[str, Any] | None = None
    scoring_post_receipt: dict[str, Any] | None = None
    service_response: dict[str, Any] | None = None
    action_image: tuple[float, float] | None = None
    guard_context: Any = None
    action_css: list[float] | None = None
    after: dict[str, Any] | None = None
    after_image: np.ndarray | None = None
    outcome: dict[str, Any] | None = None
