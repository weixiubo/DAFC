from __future__ import annotations
import copy, math
from typing import Any
from dafc.experiments.android_atomic import ACTORS


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
    audit = {
        actor: _empty_actor_audit() for actor in ACTORS
    }
    for logical_id, payload in sorted(normalized.items()):
        actor = payload.get("kind")
        if actor not in audit:
            raise RuntimeError(
                f"coordinate repair saw an unknown actor: {logical_id}"
            )
        steps = payload.get("steps")
        if not isinstance(steps, list):
            raise RuntimeError("coordinate repair saw absent episode steps")
        for step in steps:
            if not isinstance(step, dict) or step.get("action_executed") is not True:
                continue
            actor_audit = audit[actor]
            actor_audit["executed_steps"] += 1
            coordinates = step.get("action_xy")
            if not isinstance(coordinates, list) or len(coordinates) != 2:
                raise RuntimeError("coordinate repair requires an exact pair")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in coordinates
            ):
                raise RuntimeError(
                    "coordinate repair rejects nonfinite or nonnumeric values"
                )
            actor_audit["finite_numeric_pairs"] += 1
            if not all(float(value).is_integer() for value in coordinates):
                raise RuntimeError(
                    "coordinate repair rejects fractional pixel coordinates"
                )
            actor_audit["integral_value_pairs"] += 1
            if all(isinstance(value, float) for value in coordinates):
                actor_audit["original_float_pairs"] += 1
            elif all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in coordinates
            ):
                actor_audit["original_int_pairs"] += 1
            else:
                actor_audit["original_mixed_numeric_pairs"] += 1
            decision = step.get("decision")
            executed = (
                decision.get("executed_action_xy")
                if isinstance(decision, dict)
                else None
            )
            if (
                not isinstance(executed, list)
                or len(executed) != 2
                or executed != coordinates
            ):
                raise RuntimeError(
                    "coordinate repair requires the executed-action reverse binding"
                )
            actor_audit["matches_executed_decision"] += 1
            step["action_xy"] = [int(float(value)) for value in coordinates]
            actor_audit["normalized_pairs"] += 1
    return normalized, audit
