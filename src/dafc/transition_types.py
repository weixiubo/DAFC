from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


@dataclass(frozen=True)
class Transition:
    """One source transition admitted from a completed successful episode."""

    transition_id: str
    demonstration_order: int
    source_success: bool
    before: np.ndarray
    action_xy: tuple[float, float]
    after: np.ndarray
    terminal: bool = False

    def __post_init__(self) -> None:
        if not self.transition_id:
            raise ValueError("transition_id must be non-empty")
        if self.demonstration_order < 0:
            raise ValueError("demonstration_order must be non-negative")
        if self.before.ndim != 3 or self.before.shape[2] != 3:
            raise ValueError("before must be an HxWx3 image")
        if self.after.shape != self.before.shape:
            raise ValueError("before and after images must have the same shape")
        if not all((math.isfinite(value) for value in self.action_xy)):
            raise ValueError("action_xy must be finite")


@dataclass
class AuthorizationContext:
    """Non-serializable templates retained only until post-action checking."""

    transition_id: str
    demonstration_order: int
    mapped_action_xy: tuple[float, float]
    warped_before: np.ndarray
    warped_after: np.ndarray
    authorization_receipt: dict[str, Any]
    terminal: bool = False


METHOD_VERSION = "transition-guard-v2.4-progress-recovery"
MatchProvider = Callable[[np.ndarray, np.ndarray], dict[str, Any]]
