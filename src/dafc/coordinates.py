from __future__ import annotations

import copy
import math
from typing import Any

ACTORS = ("qwen25vl", "uitars")


def _empty_actor_audit() -> dict[str, int]:
    return {
        "executed_steps": 0,
        "finite_numeric_pairs": 0,
        "integral_value_pairs": 0,
        "original_float_pairs": 0,
        "original_int_pairs": 0,
        "original_mixed_numeric_pairs": 0,
        "matches_executed_decision": 0,
        "normalized_pairs": 0,
    }


def normalize_integral_action_coordinates(
    completed_payloads: dict[str, dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, int]]]:
    """Normalize only an in-memory copy; reject any non-integral coordinate."""
    normalized = copy.deepcopy(completed_payloads)
    audit = {actor: _empty_actor_audit() for actor in ACTORS}
    for logical_id, payload in sorted(normalized.items()):
        actor = payload.get("kind")
        if actor not in audit:
            raise RuntimeError(f"coordinate repair saw an unknown actor: {logical_id}")
        steps = payload.get("steps")
        if not isinstance(steps, list):
            raise RuntimeError("coordinate repair saw absent episode steps")
        for step in steps:
            if not isinstance(step, dict) or step.get("action_executed") is not True:
                continue
            _normalize_step(step, audit[actor])
    return (normalized, audit)


def _integral_pair(value: Any) -> list[int | float]:
    if not isinstance(value, list) or len(value) != 2:
        raise RuntimeError("coordinate repair requires an exact pair")
    if any(
        (
            isinstance(item, bool) or not isinstance(item, (int, float)) or (not math.isfinite(float(item)))
            for item in value
        )
    ):
        raise RuntimeError("coordinate repair rejects nonfinite or nonnumeric values")
    if not all((float(item).is_integer() for item in value)):
        raise RuntimeError("coordinate repair rejects fractional pixel coordinates")
    return value


def _pair_kind(coordinates: list[int | float]) -> str:
    if all((isinstance(value, float) for value in coordinates)):
        return "original_float_pairs"
    if all((isinstance(value, int) and (not isinstance(value, bool)) for value in coordinates)):
        return "original_int_pairs"
    return "original_mixed_numeric_pairs"


def _normalize_step(step: dict[str, Any], actor_audit: dict[str, int]) -> None:
    actor_audit["executed_steps"] += 1
    coordinates = _integral_pair(step.get("action_xy"))
    actor_audit["finite_numeric_pairs"] += 1
    actor_audit["integral_value_pairs"] += 1
    actor_audit[_pair_kind(coordinates)] += 1
    decision = step.get("decision")
    executed = decision.get("executed_action_xy") if isinstance(decision, dict) else None
    if not isinstance(executed, list) or len(executed) != 2 or executed != coordinates:
        raise RuntimeError("coordinate repair requires the executed-action reverse binding")
    actor_audit["matches_executed_decision"] += 1
    step["action_xy"] = [int(float(value)) for value in coordinates]
    actor_audit["normalized_pairs"] += 1
