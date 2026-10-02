from types import SimpleNamespace

import numpy as np
import pytest

from dafc import transition_selection as selection
from dafc.transition_types import Transition

from .frozen_function import original_function


@pytest.mark.parametrize(
    "mode",
    ["success", "hidden", "progress", "blocked", "confirmed", "risk-missing", "risk-inactive", "risk-active"],
)
@pytest.mark.parametrize("require_precondition", [True, False])
def test_prefix_progress_risk_and_receipts_match_original(mode, require_precondition, monkeypatch):
    image = np.zeros((16, 16, 3), np.uint8)
    transitions = [Transition(f"t{i}", i, True, image, (4.0, 5.0), image, terminal=i == 1) for i in range(2)]
    events = []

    def source_effect(transition):
        return {"observable": not (mode == "hidden" and transition.demonstration_order == 0)}

    def authorize(transition, current, **kwargs):
        events.append(transition.transition_id)
        if transition.demonstration_order == 0 and mode in {"progress", "blocked"}:
            return {
                "precondition": {
                    "observable": True,
                    "accepted": False,
                    "normalized_margin": -2.0 if mode == "progress" else 2.0,
                    "context_error": 0.0,
                }
            }, None
        return {"decision": "reuse", "transition_id": transition.transition_id}, SimpleNamespace(
            mapped_action_xy=(4.0, 5.0)
        )

    monkeypatch.setattr(selection, "source_effect_receipt", source_effect)
    monkeypatch.setattr(selection, "authorize", authorize)
    original = original_function("src/dafc/transitions.py", "select", dict(vars(selection)))
    certificate = (
        SimpleNamespace(active=mode == "risk-active", receipt=lambda: {"active": mode == "risk-active"})
        if mode in {"risk-inactive", "risk-active"}
        else None
    )
    options = {
        "require_precondition": require_precondition,
        "confirmed_transition_ids": ["t0"] if mode == "confirmed" else [],
        "require_risk_certificate": mode.startswith("risk"),
        "risk_certificate": certificate,
    }
    expected, _ = original(transitions, image, **options)
    expected_events = events[:]
    events.clear()
    actual, _ = selection.select(transitions, image, **options)
    assert actual == expected
    assert events == expected_events
