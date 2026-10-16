"""Recompute published statistical estimates from the supplied evidence."""

import json
from collections import defaultdict
from pathlib import Path

from dafc.evaluation.attempts import sequence_errors
from dafc.evidence_data.frozen_statistics import (
    ACTORS,
    BOOTSTRAP_SEED,
    POLICIES,
    compact_row,
    comparison,
    policy_summary,
    sha256,
)

HERE = Path(__file__).resolve().parent / "evidence_data"


def load(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def verify(raw_root=None):
    for name, expected in load(HERE / "checksums.json").items():
        if not sha256(HERE / name) == expected:
            raise ValueError(f"publication content changed: {name}")
    rows = load(HERE / "rows.json")
    expected = load(HERE / "expected.json")
    _verify_row_invariants(rows)
    actual = {a: {p: policy_summary(rows, a, p) for p in POLICIES} for a in ACTORS}
    if not actual == expected["summary"]:
        raise ValueError("policy summaries changed")
    comparisons = {}
    seed = BOOTSTRAP_SEED
    for actor in ACTORS:
        comparisons[actor] = {}
        for right in ("always", "s_only", "standard_ransac"):
            comparisons[actor]["dafc_minus_" + right] = comparison(rows, actor, "dafc", right, seed)
            seed += 1
    if not comparisons == expected["comparisons"]:
        raise ValueError("paired estimates or bootstrap intervals changed")
    if raw_root is not None:
        _verify_raw(raw_root, rows)
    return {
        "status": "verified",
        "outcomes": 1120,
        "policy_summaries": 8,
        "paired_bootstrap_comparisons": 6,
        "bootstrap_samples": 20000,
        "raw_evidence_verified": raw_root is not None,
        "historical_result": True,
        "evidence_edition": "public-2026-10-17",
        "model_message_bodies_included": False,
    }


def _verify_raw(raw_root, rows):
    root = Path(raw_root).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Supply the published C587-v3 attempt directory: {root}")
    files = load(HERE / "raw_manifest.json")["files"]
    for record in files:
        path = (root / record["path"]).resolve()
        if not (root in path.parents and sha256(path) == record["sha256"]):
            raise ValueError(record["path"])
    lifecycle = defaultdict(list)
    outcomes = {}
    for path in sorted((root / "results").rglob("*.json")):
        payload = load(path)
        lifecycle[payload["job"]["job_id"]].append(payload)
        _record_outcome(payload, path, root, outcomes)
        _verify_images(payload, root)
    if not len(lifecycle) == 1120:
        raise ValueError("Frozen evidence invariant failed")
    for attempts in lifecycle.values():
        if sequence_errors(sorted(attempts, key=lambda x: x["attempt"]), require_complete=True):
            raise ValueError("Frozen evidence invariant failed")
    if not outcomes == {r["job_id"]: r for r in rows}:
        raise ValueError("raw-to-public projection changed")


def _verify_images(payload, root):
    for key in ("before", "before_raw", "after"):
        image = payload.get(key)
        if not image:
            continue
        parts = Path(image["path"]).parts
        pos = parts.index("images")
        image_path = root / Path(*parts[pos:])
        if not sha256(image_path) == image["sha256"]:
            raise ValueError(f"image receipt mismatch: {image_path}")


def _record_outcome(payload, path, root, outcomes):
    if payload.get("outcome_bearing"):
        item = compact_row(payload, path)
        item["attempt_path"] = str(path.relative_to(root))
        if not item["job_id"] not in outcomes:
            raise ValueError("Frozen evidence invariant failed")
        outcomes[item["job_id"]] = item


def _verify_row_invariants(rows):
    if not (len(rows) == 1120 and len({r["job_id"] for r in rows}) == 1120):
        raise ValueError("Frozen evidence invariant failed")
    if not all((not r["authorized"] for r in rows if not r["source_success"])):
        raise ValueError("Frozen evidence invariant failed")
    if not all((r["model_calls"] == 1 for r in rows if r["policy"] == "always")):
        raise ValueError("Frozen evidence invariant failed")
