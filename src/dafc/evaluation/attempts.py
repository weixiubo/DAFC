from __future__ import annotations

from typing import Any


def sequence_errors(attempts: list[dict[str, Any]], require_complete: bool) -> list[str]:
    """Validate one job's immutable at-most-two-attempt lifecycle."""
    errors: list[str] = []
    indices = [row.get("attempt") for row in attempts]
    if indices not in ([], [0], [0, 1]):
        errors.append(f"invalid attempt indices/order: {indices}")
        return errors
    outcomes = [row for row in attempts if row.get("outcome_bearing") is True]
    if len(outcomes) > 1:
        errors.append("more than one outcome-bearing attempt")
    if attempts:
        _retention_errors(attempts, errors)
    if require_complete and len(outcomes) != 1:
        errors.append("job lacks exactly one outcome-bearing attempt")
    return errors


def _retention_errors(attempts, errors):
    first = attempts[0]
    if first.get("outcome_bearing") is True:
        if len(attempts) != 1:
            errors.append("outcome-bearing attempt was retried")
        if first.get("retry_permitted") is not False:
            errors.append("outcome-bearing attempt permits retry")
    else:
        if first.get("status") != "infrastructure_failure":
            errors.append("non-outcome attempt is not infrastructure_failure")
        if first.get("retry_permitted") is not True:
            errors.append("first non-outcome attempt does not permit its one retry")
        if len(attempts) == 2:
            second = attempts[1]
            if second.get("outcome_bearing") is not True:
                errors.append("second and final attempt is not outcome-bearing")
            if second.get("retry_permitted") is not False:
                errors.append("second attempt permits another retry")
