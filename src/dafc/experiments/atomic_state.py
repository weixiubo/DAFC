from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from dafc.adapters.android.device import AndroidDevice
from dafc.experiments.atomic_design import AtomicTask
from dafc.experiments.atomic_design import jobs_for_actor as jobs_for_actor
from dafc.experiments.attempts import next_attempt as next_attempt
from dafc.routing import STANDARD_MIN_INLIERS as STANDARD_MIN_INLIERS
from dafc.routing import standard_registration as standard_registration


@dataclass(frozen=True, kw_only=True)
class TrialConfig:
    env: Any
    task_registry: dict[str, Any]
    manifest: dict[str, Any]
    manifest_path: Path
    actor: str
    service_url: str
    root: Path
    job: dict[str, Any]
    attempt: int
    device: AndroidDevice


@dataclass
class TrialState:
    spec: AtomicTask | None = None
    source: dict[str, Any] | None = None
    path: Path | None = None
    image_dir: Path | None = None
    stage: str = "start"
    task: Any = None
    model_calls: int = 0
    authorization_started: bool = False
    target_action_started: bool = False
    started: float = 0.0
    goal: str | None = None
    raw: np.ndarray | None = None
    observed: np.ndarray | None = None
    setup_actions: list[dict[str, Any]] | None = None
    ui_receipts: list[dict[str, Any]] | None = None
    before: dict[str, Any] | None = None
    raw_before: dict[str, Any] | None = None
    authorization: dict[str, Any] | None = None
    model_response: dict[str, Any] | None = None
    requested_action: list[float] | None = None
    action_source: str | None = None
    after: dict[str, Any] | None = None
    result: dict[str, Any] | None = None
