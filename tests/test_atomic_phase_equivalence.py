"""Preserve event order, budgets, retry classification and cleanup of original atomic jobs."""

from types import SimpleNamespace

import numpy as np
import pytest

from dafc.experiments import android_atomic as runner
from dafc.experiments import atomic_steps as steps
from dafc.experiments.atomic_design import TASKS
from dafc.storage import write_new_json

from .episode_harness import normalized
from .frozen_function import original_function


def exercise(root, monkeypatch, policy, failure, *, original):
    events = []
    image = np.zeros((32, 40, 3), np.uint8)

    def event(name):
        events.append(name)
        if failure == name:
            raise RuntimeError(name)

    task = SimpleNamespace(tear_down=lambda env: event("cleanup"), is_successful=lambda env: 1.0)

    def setup(*args, **kwargs):
        event("setup")
        return task, "goal", image, image, [], [{"clean": True}]

    def authorize(*args):
        event("authorize")
        return {
            "accepted": policy != "always",
            "mapped_action_xy": [12.0, 13.0] if policy != "always" else None,
        }

    def infer(*args, **kwargs):
        event("model")
        return {"result": {"parsed_action": {"valid": failure != "parse", "normalized_xy": [0.3, 0.4]}}}

    def execute(*args):
        event("action")
        return [12.0, 13.0]

    def clean(*args, **kwargs):
        event("observe-after")
        return {"clean": True}

    def restore():
        event("restore")

    monkeypatch.setattr(steps, "setup_target_state", setup)
    monkeypatch.setattr(steps, "authorize", authorize)
    monkeypatch.setattr(steps.GROUND_CLIENT, "infer", infer)
    monkeypatch.setattr(steps, "compact_model_response", lambda response: response)
    monkeypatch.setattr(steps, "execute_click", execute)
    monkeypatch.setattr(steps, "ensure_clean_observation_ui", clean)
    monkeypatch.setattr(steps.C482, "restore_display", restore, raising=False)
    monkeypatch.setattr(steps.C482, "pixels", lambda state: image)
    monkeypatch.setattr(steps.C482, "observe", lambda array, contract: array)
    manifest = {
        "jobs_sha256": "design",
        "design": {"tasks": [{"task_id": TASKS[0].task_id, "source": {"source_success": True}}]},
    }
    path = root / "manifest.json"
    write_new_json(path, manifest)
    arguments = dict(
        env=SimpleNamespace(get_state=lambda **kwargs: image),
        task_registry={},
        manifest=manifest,
        manifest_path=path,
        actor="qwen25vl",
        service_url="http://127.0.0.1:1",
        root=root,
        job={"task": TASKS[0].task_id, "contract": "identity", "job_id": "test", "policy": policy},
        attempt=0,
    )
    namespace = dict(vars(steps))
    namespace.update(append_ledger=runner.append_ledger, write_new_json=write_new_json)
    if original:
        call = original_function("src/dafc/experiments/android_atomic.py", "run_one", namespace)
    else:
        call = runner.run_one
        arguments["device"] = SimpleNamespace(restore_display=restore)
    payload = call(**arguments)
    assert payload["paper_evidence"] is original
    # Scientific receipts and event ordering remain exact. New runs explicitly lose historical identity.
    payload = {
        key: value for key, value in payload.items() if key not in {"paper_evidence", "runner_version"}
    }
    return normalized(payload, root), events


@pytest.mark.parametrize(
    "policy,failure",
    [
        ("always", None),
        ("dafc", None),
        ("always", "setup"),
        ("dafc", "authorize"),
        ("always", "model"),
        ("always", "parse"),
        ("dafc", "action"),
        ("always", "observe-after"),
        ("always", "cleanup"),
        ("always", "restore"),
    ],
)
def test_atomic_phases_preserve_original(policy, failure, tmp_path, monkeypatch):
    with monkeypatch.context() as context:
        expected = exercise(tmp_path / "old", context, policy, failure, original=True)
    with monkeypatch.context() as context:
        actual = exercise(tmp_path / "new", context, policy, failure, original=False)
    assert actual == expected
