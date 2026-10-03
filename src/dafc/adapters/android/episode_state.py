from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

import dafc.transitions as GUARD
from dafc.adapters.android.device import AndroidDevice


@dataclass(frozen=True, kw_only=True)
class EpisodeConfig:
    env: Any
    task_registry: dict[str, Any]
    task_id: str
    contract_id: str
    policy: str
    kind: str
    model_service_url: str
    run_id: str
    attempt: int
    output_root: Path
    timer_params: dict[str, Any] | None = None
    browser_task_seed: Any | None = None
    memory_goal: str | None = None
    memory_params: dict[str, Any] | None = None
    memory: list[GUARD.Transition] | None = None
    match_provider: Callable[[np.ndarray, np.ndarray], dict[str, Any]] | None = None
    risk_certificate: GUARD.RiskCertificate | None = None
    require_risk_certificate: bool = False
    score_postcondition_for_ablations: bool = False
    device: AndroidDevice


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
    spec: Any = None
    contract: Any = None
    params_fingerprint: str = ""
    task: Any = None
    history_actions: list[dict[str, Any]] = field(default_factory=list)
    initial_reward: float | None = None
    final_reward: float = 0.0
    observation_ui_receipts: list[dict[str, Any]] = field(default_factory=list)
    current: np.ndarray | None = None


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
    before_receipt: dict[str, Any] = field(default_factory=dict)
    context: Any = None
    model_response: dict[str, Any] | None = None
    action_xy: tuple[float, float] | None = None
    next_observation: np.ndarray | None = None
    after_receipt: dict[str, Any] | None = None
    post: dict[str, Any] | None = None
    scoring_post: dict[str, Any] | None = None
