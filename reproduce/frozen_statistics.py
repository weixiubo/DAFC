"""Unchanged C588 statistical function bodies; portable imports/constants only."""
from __future__ import annotations
import hashlib
from pathlib import Path
from typing import Any
import numpy as np
BOOTSTRAP_REPLICATES = 20_000
BOOTSTRAP_SEED = 588001
POLICIES = ("always", "s_only", "standard_ransac", "dafc")
ACTORS = ("qwen25vl", "uitars")
def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

def compact_row(payload: dict[str, Any], path: Path) -> dict[str, Any]:
    job = payload["job"]
    authorization = payload.get("authorization") or {}
    return {
        "actor": job["actor"],
        "task": job["task"],
        "contract": job["contract"],
        "policy": job["policy"],
        "repetition": int(job["repetition"]),
        "job_id": job["job_id"],
        "attempt": int(payload["attempt"]),
        "attempt_path": str(path.resolve()),
        "attempt_sha256": sha256(path),
        "status": payload["status"],
        "success": bool(payload.get("success")),
        "official_reward": payload.get("official_reward"),
        "model_calls": int(payload.get("model_calls", 0)),
        "action_executed": bool(payload.get("action_executed")),
        "action_source": payload.get("action_source"),
        "authorized": bool(authorization.get("accepted")),
        "authorization_reason": authorization.get("reason"),
        "source_success": bool(payload.get("source", {}).get("source_success")),
        "scientific_failure": payload.get("scientific_failure"),
        "observation_ui_clean": bool(
            payload.get("observation_ui", {}).get("all_clean", False)
        ),
    }

def policy_summary(rows: list[dict[str, Any]], actor: str, policy: str) -> dict[str, Any]:
    selected = [
        row for row in rows if row["actor"] == actor and row["policy"] == policy
    ]
    if len(selected) != 140:
        raise RuntimeError(f"unexpected actor/policy rows: {actor}/{policy}")
    by_task = {}
    for task in sorted({row["task"] for row in selected}):
        subset = [row for row in selected if row["task"] == task]
        by_task[task] = {
            "episodes": len(subset),
            "successes": sum(row["success"] for row in subset),
            "success_rate": np.mean([row["success"] for row in subset]),
            "model_calls": sum(row["model_calls"] for row in subset),
            "authorization_rate": np.mean([row["authorized"] for row in subset]),
            "source_success": subset[0]["source_success"],
        }
    by_contract = {}
    for contract in sorted({row["contract"] for row in selected}):
        subset = [row for row in selected if row["contract"] == contract]
        by_contract[contract] = {
            "episodes": len(subset),
            "successes": sum(row["success"] for row in subset),
            "success_rate": np.mean([row["success"] for row in subset]),
            "model_calls": sum(row["model_calls"] for row in subset),
            "authorization_rate": np.mean([row["authorized"] for row in subset]),
        }
    authorized = [row for row in selected if row["authorized"]]
    return {
        "episodes": len(selected),
        "successes": sum(row["success"] for row in selected),
        "success_rate": float(np.mean([row["success"] for row in selected])),
        "task_equal_success_rate": float(
            np.mean([value["success_rate"] for value in by_task.values()])
        ),
        "model_calls": sum(row["model_calls"] for row in selected),
        "target_call_rate": float(
            sum(row["model_calls"] for row in selected) / len(selected)
        ),
        "target_call_reduction": float(
            1.0 - sum(row["model_calls"] for row in selected) / len(selected)
        ),
        "authorizations": len(authorized),
        "authorization_rate": len(authorized) / len(selected),
        "authorized_successes": sum(row["success"] for row in authorized),
        "authorized_success_rate": (
            sum(row["success"] for row in authorized) / len(authorized)
            if authorized
            else None
        ),
        "by_task": by_task,
        "by_contract": by_contract,
    }

def pair_map(rows: list[dict[str, Any]], actor: str, policy: str) -> dict[tuple[str, str, int], dict[str, Any]]:
    return {
        (row["task"], row["contract"], row["repetition"]): row
        for row in rows
        if row["actor"] == actor and row["policy"] == policy
    }

def cluster_bootstrap(
    pairs: list[tuple[str, int]], seed: int
) -> dict[str, Any]:
    grouped: dict[str, list[int]] = {}
    for task, difference in pairs:
        grouped.setdefault(task, []).append(difference)
    tasks = sorted(grouped)
    arrays = [np.asarray(grouped[task], dtype=np.float64) for task in tasks]
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=np.float64)
    for index in range(BOOTSTRAP_REPLICATES):
        selected = rng.integers(0, len(arrays), size=len(arrays))
        values = np.concatenate([arrays[int(position)] for position in selected])
        draws[index] = float(values.mean())
    observed = float(np.mean([difference for _, difference in pairs]))
    return {
        "cluster_unit": "task",
        "clusters": len(tasks),
        "replicates": BOOTSTRAP_REPLICATES,
        "rng_seed": seed,
        "observed_difference": observed,
        "ci95": [
            float(np.quantile(draws, 0.025)),
            float(np.quantile(draws, 0.975)),
        ],
    }

def comparison(
    rows: list[dict[str, Any]], actor: str, left: str, right: str, seed: int
) -> dict[str, Any]:
    left_rows = pair_map(rows, actor, left)
    right_rows = pair_map(rows, actor, right)
    if set(left_rows) != set(right_rows) or len(left_rows) != 140:
        raise RuntimeError("paired policy keys changed")
    paired = [
        (key, bool(left_rows[key]["success"]), bool(right_rows[key]["success"]))
        for key in sorted(left_rows)
    ]
    differences = [(key[0], int(a) - int(b)) for key, a, b in paired]
    return {
        "left": left,
        "right": right,
        "pairs": len(paired),
        "both_success": sum(a and b for _, a, b in paired),
        "left_only_success": sum(a and not b for _, a, b in paired),
        "right_only_success": sum(not a and b for _, a, b in paired),
        "both_failure": sum(not a and not b for _, a, b in paired),
        "success_difference": float(np.mean([value for _, value in differences])),
        "task_cluster_bootstrap": cluster_bootstrap(differences, seed),
    }
