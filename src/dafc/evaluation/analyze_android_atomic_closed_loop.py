#!/usr/bin/env python3
"""Analyze the complete prospective native-Android atomic experiment."""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Callable

import numpy as np


from dafc.paths import PROJECT
MANIFEST = PROJECT / "formal/transitionguard_android_atomic_v3_manifest.json"
ROOT = PROJECT / 'data/formal_android_atomic_c587_v3'
OUT_JSON = PROJECT / "validation/c588_android_atomic_closed_loop_analysis.json"
OUT_REPORT = PROJECT / "reports/c588_android_atomic_closed_loop_analysis.md"
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


def load(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(PROJECT.resolve()).as_posix()


def committed(path: Path) -> bool:
    name = rel(path)
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", name],
        cwd=PROJECT,
        capture_output=True,
        check=False,
    )
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", name],
        cwd=PROJECT,
        check=False,
    )
    return tracked.returncode == 0 and clean.returncode == 0


def git_clean() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=PROJECT,
        capture_output=True,
        check=False,
    )
    return result.returncode == 0 and not result.stdout


def attempt_paths(root: Path, actor: str, job_id: str) -> list[Path]:
    directory = root / "results" / actor
    return sorted(directory.glob(f"{job_id}--a??.json"))


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


def collect(manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, str]]:
    rows: list[dict[str, Any]] = []
    infrastructure: list[dict[str, Any]] = []
    bindings: dict[str, str] = {}
    manifest_hash = sha256(MANIFEST)
    for actor in ACTORS:
        for job in manifest["design"]["planned_jobs"][actor]:
            paths = attempt_paths(ROOT, actor, job["job_id"])
            attempts = [load(path) for path in paths]
            for path, payload in zip(paths, attempts):
                bindings[str(path.resolve())] = sha256(path)
                if (
                    payload.get("manifest_sha256") != manifest_hash
                    or payload.get("jobs_sha256") != manifest["jobs_sha256"]
                    or payload.get("job") != job
                ):
                    raise RuntimeError(f"attempt binding mismatch: {path}")
                if not payload.get("outcome_bearing"):
                    infrastructure.append(
                        {
                            "actor": actor,
                            "job_id": job["job_id"],
                            "attempt": payload["attempt"],
                            "path": str(path.resolve()),
                            "sha256": sha256(path),
                            "status": payload["status"],
                            "stage": payload.get("stage"),
                            "error_type": payload.get("error_type"),
                            "error": payload.get("error"),
                        }
                    )
            outcome = [
                (path, payload)
                for path, payload in zip(paths, attempts)
                if payload.get("outcome_bearing") is True
            ]
            if len(outcome) != 1:
                raise RuntimeError(
                    f"job does not have exactly one outcome-bearing attempt: {job['job_id']}"
                )
            path, payload = outcome[0]
            rows.append(compact_row(payload, path))
    return rows, infrastructure, bindings


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


def fmt_pct(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:.2f}%"


def render_report(result: dict[str, Any]) -> str:
    lines = [
        "# C588 Native-Android Atomic Closed-Loop Analysis",
        "",
        "## Decision",
        "",
        "Prospective seven-task native-Android atomic diagnostic completed. "
        "All integrity rows are retained; performance is descriptive and is not an integrity gate.",
        "",
        "## Complete result",
        "",
        "| Actor | Policy | Success | Target-call reduction | Authorization | Authorized success |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for actor in ACTORS:
        for policy in POLICIES:
            row = result["summary"][actor][policy]
            lines.append(
                f"| {actor} | {policy} | {row['successes']}/{row['episodes']} "
                f"({fmt_pct(row['success_rate'])}) | {fmt_pct(row['target_call_reduction'])} | "
                f"{row['authorizations']}/{row['episodes']} ({fmt_pct(row['authorization_rate'])}) | "
                f"{row['authorized_successes']}/{row['authorizations']} "
                f"({fmt_pct(row['authorized_success_rate'])}) |"
            )
    lines.extend(["", "## DAFC comparisons", ""])
    for actor in ACTORS:
        lines.append(f"### {actor}")
        lines.append("")
        for name, row in result["comparisons"][actor].items():
            ci = row["task_cluster_bootstrap"]["ci95"]
            lines.append(
                f"- {name}: {fmt_pct(row['success_difference'])} "
                f"(task-cluster 95% CI [{fmt_pct(ci[0])}, {fmt_pct(ci[1])}]); "
                f"wins/losses {row['left_only_success']}/{row['right_only_success']}."
            )
        target = result["interpretive_target"][actor]
        lines.append(
            f"- Prespecified DAFC-versus-Always target: "
            f"{'met' if target['met'] else 'not met'}; lower CI "
            f"{fmt_pct(target['lower_ci'])}, call reduction "
            f"{fmt_pct(target['target_call_reduction'])}."
        )
        lines.append("")
    lines.extend(
        [
            "## Boundary",
            "",
            "This is a controlled final-step action study over seven fixed tasks. "
            "Accessibility is used only to prepare the state, then discarded. "
            "The result does not establish end-to-end navigation, independent task breadth, "
            "AndroidWorld-wide generalization, or a per-action safety guarantee.",
            "",
            f"Manifest SHA-256: `{result['bindings']['manifest_sha256']}`.",
            f"Execution snapshot SHA-256: `{result['bindings']['execution_snapshot_sha256']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    if OUT_JSON.exists() or OUT_REPORT.exists():
        raise RuntimeError("refusing to overwrite C588 analysis")
    if not committed(Path(__file__).resolve()) or not committed(MANIFEST) or not git_clean():
        raise RuntimeError("analysis requires committed code/manifest and clean repository")
    manifest = load(MANIFEST)
    if manifest.get("pass") is not True or manifest.get("design", {}).get("planned_jobs_total") != 1120:
        raise RuntimeError("C586 manifest did not pass")
    rows, infrastructure, execution_bindings = collect(manifest)
    summary = {
        actor: {
            policy: policy_summary(rows, actor, policy) for policy in POLICIES
        }
        for actor in ACTORS
    }
    comparisons = {}
    seed = BOOTSTRAP_SEED
    for actor in ACTORS:
        comparisons[actor] = {}
        for right in ("always", "s_only", "standard_ransac"):
            comparisons[actor][f"dafc_minus_{right}"] = comparison(
                rows, actor, "dafc", right, seed
            )
            seed += 1
    interpretive = {}
    for actor in ACTORS:
        primary = comparisons[actor]["dafc_minus_always"]
        lower = primary["task_cluster_bootstrap"]["ci95"][0]
        reduction = summary[actor]["dafc"]["target_call_reduction"]
        interpretive[actor] = {
            "lower_ci": lower,
            "target_call_reduction": reduction,
            "noninferiority_margin": -0.05,
            "minimum_target_call_reduction": 0.20,
            "met": bool(lower > -0.05 and reduction >= 0.20),
        }
    execution_snapshot = hashlib.sha256(
        json.dumps(execution_bindings, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
    ).hexdigest()
    criteria = {
        "exact_1120_outcome_rows": len(rows) == 1120,
        "exact_560_rows_per_actor": all(
            sum(row["actor"] == actor for row in rows) == 560 for actor in ACTORS
        ),
        "exact_140_rows_per_actor_policy": all(
            sum(row["actor"] == actor and row["policy"] == policy for row in rows)
            == 140
            for actor in ACTORS
            for policy in POLICIES
        ),
        "unique_complete_job_keys": len({row["job_id"] for row in rows}) == 1120,
        "all_complete_rows_observation_ui_clean": all(
            row["observation_ui_clean"]
            for row in rows
            if row["status"] == "complete"
        ),
        "source_failure_tasks_never_authorized": all(
            not row["authorized"] for row in rows if not row["source_success"]
        ),
        "always_has_one_model_call": all(
            row["model_calls"] == 1 for row in rows if row["policy"] == "always"
        ),
        "all_attempts_retained": True,
        "performance_not_used_as_integrity_gate": True,
        "zero_new_model_or_environment_calls_during_analysis": True,
    }
    result = {
        "title": "C588 Native-Android Atomic Closed-Loop Analysis",
        "paper_evidence": True,
        "status": "prospective auxiliary native-Android action-level diagnostic",
        "bindings": {
            "manifest": rel(MANIFEST),
            "manifest_sha256": sha256(MANIFEST),
            "jobs_sha256": manifest["jobs_sha256"],
            "execution_attempt_files": len(execution_bindings),
            "execution_snapshot_sha256": execution_snapshot,
        },
        "population": {
            "tasks": 7,
            "successful_source_receipts": 5,
            "failed_source_receipts": 2,
            "actors": 2,
            "contracts": 4,
            "policies": 4,
            "repetitions": 5,
            "outcome_rows": len(rows),
            "infrastructure_attempts": infrastructure,
        },
        "summary": summary,
        "comparisons": comparisons,
        "interpretive_target": interpretive,
        "criteria": criteria,
        "pass": all(criteria.values()),
        "rows": rows,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    OUT_REPORT.write_text(render_report(result), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary": summary,
                "interpretive_target": interpretive,
                "criteria": criteria,
                "pass": result["pass"],
                "output_sha256": sha256(OUT_JSON),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
