"""Ledger, design, screenshot and in-flight checks protect interrupted device runs."""

import json

import pytest

from dafc.experiments.attempts import append_ledger, result_path
from dafc.jobs.atomic import verify_recovery
from dafc.resources import sha256_file
from dafc.storage import write_new_json


@pytest.fixture
def durable(tmp_path):
    job = {"job_id": "run-job", "task": "task", "policy": "always"}
    write_new_json(tmp_path / "experiment.json", {"design": [job]})
    image = tmp_path / "observed.png"
    image.write_bytes(b"original screenshot bytes")
    payload = {
        "job": job,
        "attempt": 0,
        "outcome_bearing": True,
        "retry_permitted": False,
        "status": "complete",
        "manifest_sha256": sha256_file(tmp_path / "experiment.json"),
        "before": {"path": str(image), "sha256": sha256_file(image)},
    }
    path = result_path(tmp_path, "qwen25vl", job, 0)
    write_new_json(path, payload)
    ledger = {
        "attempt_path": str(path.resolve()),
        "attempt_sha256": sha256_file(path),
        "job_id": job["job_id"],
        "attempt": 0,
    }
    append_ledger(tmp_path, "qwen25vl", ledger)
    return tmp_path, job, payload, ledger


def test_durable_attempt_clears_only_matching_inflight_marker(durable):
    root, job, _, _ = durable
    write_new_json(root / "inflight.json", {"job": job, "attempt": 0})
    verify_recovery(root, "qwen25vl", [job])
    assert not (root / "inflight.json").exists()


@pytest.mark.parametrize(
    "damage", ["attempt", "image", "manifest", "duplicate", "identity", "missing_ledger"]
)
def test_tampered_recovery_is_rejected(durable, damage):
    root, job, payload, ledger = durable
    if damage == "attempt":
        result_path(root, "qwen25vl", job, 0).write_text("{}")
    elif damage == "image":
        (root / "observed.png").write_bytes(b"changed")
    elif damage == "manifest":
        (root / "experiment.json").write_text("{}")
    elif damage == "duplicate":
        append_ledger(root, "qwen25vl", ledger)
    elif damage == "identity":
        ledger["job_id"] = "unexpected"
        (root / "ledger/qwen25vl.jsonl").write_text(json.dumps(ledger) + "\n")
    else:
        (root / "ledger/qwen25vl.jsonl").unlink()
    with pytest.raises(ValueError):
        verify_recovery(root, "qwen25vl", [job])


def test_ambiguous_action_never_automatically_retried(tmp_path):
    job = {"job_id": "uncertain"}
    write_new_json(tmp_path / "inflight.json", {"job": job, "attempt": 0})
    with pytest.raises(ValueError, match="may have executed an action"):
        verify_recovery(tmp_path, "qwen25vl", [job])
    assert (tmp_path / "inflight.json").exists()
