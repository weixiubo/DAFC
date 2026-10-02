"""Ordered transition selection with explicit confirmation and prefix state."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

import numpy as np

from dafc.risk import RiskCertificate
from dafc.transition_authorization import (
    authorize,
    confidently_observed_post_state,
    reject_receipt,
    source_effect_receipt,
)
from dafc.transition_types import METHOD_VERSION, AuthorizationContext, MatchProvider, Transition


@dataclass
class SelectionState:
    confirmed: set[str]
    progress_recovered: set[str] = field(default_factory=set)
    observable_prefix: bool = True
    replayable_predecessors: set[str] = field(default_factory=set)
    evaluations: list[dict[str, Any]] = field(default_factory=list)


def _prefix_rejection(transition, source_effect, state, require_precondition):
    observable = bool(source_effect["observable"])
    terminal_missing = bool(
        require_precondition
        and transition.terminal
        and not state.replayable_predecessors.issubset(state.confirmed)
    )
    if require_precondition and not state.observable_prefix:
        return "P_pre", "unobservable_source_predecessor"
    if terminal_missing:
        return "H", "terminal_transition_requires_confirmed_prefix"
    if require_precondition and not observable:
        return "P_pre", "unobservable_source_effect"
    if transition.transition_id in state.confirmed:
        return "H", "transition_already_post_verified"
    return None


def _evaluate(transition, current, state, require_precondition, match_provider):
    source_effect = source_effect_receipt(transition)
    rejection = _prefix_rejection(transition, source_effect, state, require_precondition)
    if rejection is not None:
        gate, reason = rejection
        precondition = source_effect if require_precondition else None
        if gate == "P_pre":
            precondition = {**source_effect, "accepted": False, "reason": reason}
        return (
            reject_receipt(transition, gate, reason, precondition=precondition),
            None,
            bool(source_effect["observable"]),
        )
    receipt, context = authorize(
        transition, current, require_precondition=require_precondition, match_provider=match_provider
    )
    if (
        require_precondition
        and not transition.terminal
        and context is None
        and confidently_observed_post_state(receipt.get("precondition"))
    ):
        receipt = {
            **receipt,
            "decision": "advance_progress",
            "failed_gate": None,
            "reason": "confident_post_state_observed",
            "progress_recovered": True,
        }
        state.confirmed.add(transition.transition_id)
        state.progress_recovered.add(transition.transition_id)
    return receipt, context, bool(source_effect["observable"])


def _selection_receipt(state, risk_control, transition=None, context=None):
    return {
        "method_version": METHOD_VERSION,
        "decision": "reuse" if context is not None else "fallback",
        "selected_transition_id": transition.transition_id if context is not None else None,
        "selected_demonstration_order": transition.demonstration_order if context is not None else None,
        "mapped_action_xy": list(context.mapped_action_xy) if context is not None else None,
        "confirmed_transition_ids": sorted(state.confirmed),
        "progress_recovered_transition_ids": sorted(state.progress_recovered),
        "observable_prefix_intact": state.observable_prefix,
        "evaluations": state.evaluations,
        "risk_control": risk_control,
    }


def select(
    transitions: Iterable[Transition],
    current: np.ndarray,
    *,
    require_precondition: bool = True,
    confirmed_transition_ids: Iterable[str] = (),
    match_provider: MatchProvider | None = None,
    risk_certificate: RiskCertificate | None = None,
    require_risk_certificate: bool = False,
) -> tuple[dict[str, Any], AuthorizationContext | None]:
    """Select the earliest applicable transition without crossing an unobserved prefix."""
    ordered = sorted(transitions, key=lambda row: row.demonstration_order)
    known_ids = {row.transition_id for row in ordered}
    if len(known_ids) != len(ordered):
        raise ValueError("transition IDs must be unique")
    state = SelectionState(set(confirmed_transition_ids))
    unknown = state.confirmed - known_ids
    if unknown:
        raise ValueError(f"confirmed transition IDs are absent from memory: {unknown}")
    risk_control = risk_certificate.receipt() if risk_certificate is not None else None
    if require_risk_certificate and (risk_certificate is None or not risk_certificate.active):
        receipt = _selection_receipt(state, risk_control)
        receipt.update(
            observable_prefix_intact=None,
            fallback_reason="risk_certificate_missing"
            if risk_certificate is None
            else "risk_certificate_inactive",
        )
        return receipt, None
    for transition in ordered:
        receipt, context, observable = _evaluate(
            transition, current, state, require_precondition, match_provider
        )
        state.evaluations.append(receipt)
        if require_precondition and not observable:
            state.observable_prefix = False
        if observable and not transition.terminal:
            state.replayable_predecessors.add(transition.transition_id)
        if context is not None:
            return _selection_receipt(state, risk_control, transition, context), context
    return _selection_receipt(state, risk_control), None
