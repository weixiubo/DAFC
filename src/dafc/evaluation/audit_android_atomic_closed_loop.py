#!/usr/bin/env python3
"""Read-only integrity audit for the C587 native-Android atomic run.

This audit is deliberately separate from the preregistered C588 performance
analysis.  It verifies the frozen job plan, attempt lifecycle, append-only
ledgers, policy information flow, and image receipts without making a model or
environment call and without using performance as a retention gate.
"""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import argparse
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


from dafc.paths import PROJECT
DEFAULT_MANIFEST = PROJECT / "formal/transitionguard_android_atomic_v3_manifest.json"
DEFAULT_ROOT = PROJECT / 'data/formal_android_atomic_c587_v3'
FINAL_OUTPUT = PROJECT / "validation/c589_android_atomic_completion_audit.json"
EXPECTED_EXECUTION_MANIFESTS = (
    Path(
        ""
        "formal/transitionguard_android_atomic_v3_manifest.json"
    ),
    Path(
        ""
        "formal/transitionguard_android_atomic_v3_manifest.json"
    ),
)
EXPECTED_MANIFEST_SHA256 = "37d35cf440725c9c7b63163f1fd8bc24cd598efeda7a222bbb658043d27b61fd"
EXPECTED_JOBS_SHA256 = "22dcd1c23b20983a7300ae4fa0e52ef4e495a592f1ddb231d66d9a21c51f9574"
EXPECTED_ACTORS = ("qwen25vl", "uitars")
EXPECTED_POLICIES = ("always", "s_only", "standard_ransac", "dafc")
EXPECTED_CONTRACTS = ("identity", "jpeg-70", "density-480", "font-1.15")
EXPECTED_TASKS = (
    "ClockStopWatchRunning",
    "ClockTimerEntry",
    "SystemWifiTurnOn",
    "SystemBluetoothTurnOn",
    "SystemWifiTurnOff",
    "SystemBluetoothTurnOff",
    "AudioRecorderRecordAudio",
)
EXPECTED_RUNNER_VERSION = "android-atomic-closed-loop-runner-v3-runtime-asset-bound"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def within(path: Path, parent: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(parent).resolve())
    except ValueError:
        return False
    return True


def add_error(errors: list[str], condition: bool, message: str) -> None:
    if not condition:
        errors.append(message)


def expected_execution_manifest_path(value: Any) -> bool:
    """Accept only the two exact paths used by the frozen C587 runner."""

    return isinstance(value, str) and value in {
        str(path) for path in EXPECTED_EXECUTION_MANIFESTS
    }


def committed_and_clean(path: Path) -> bool:
    relative = Path(path).resolve().relative_to(PROJECT.resolve()).as_posix()
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", relative],
        cwd=PROJECT,
        capture_output=True,
        check=False,
    )
    changed = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", relative],
        cwd=PROJECT,
        check=False,
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=PROJECT,
        capture_output=True,
        check=False,
    )
    return (
        tracked.returncode == 0
        and changed.returncode == 0
        and dirty.returncode == 0
        and not dirty.stdout
    )


from dafc.evaluation.attempts import (
    sequence_errors as sequence_errors,
)


def verify_image_receipt(
    receipt: dict[str, Any], expected_dir: Path, errors: list[str], label: str
) -> Path | None:
    try:
        path = Path(receipt["path"]).resolve()
    except Exception as caught:
        errors.append(f"{label}: invalid image path: {caught}")
        return None
    if not within(path, expected_dir) or path.parent != expected_dir.resolve():
        errors.append(f"{label}: image escapes exact attempt directory: {path}")
        return None
    if not path.is_file():
        errors.append(f"{label}: image is missing: {path}")
        return None
    actual_hash = sha256(path)
    if receipt.get("sha256") != actual_hash:
        errors.append(f"{label}: image hash mismatch: {path}")
    try:
        with Image.open(path) as image:
            width, height = image.size
            image.verify()
        if receipt.get("width") != width or receipt.get("height") != height:
            errors.append(f"{label}: image dimensions mismatch: {path}")
    except Exception as caught:
        errors.append(f"{label}: unreadable image: {path}: {caught}")
    return path


def validate_complete_payload(
    payload: dict[str, Any],
    expected_source: dict[str, Any],
    attempt_dir: Path,
    errors: list[str],
    prefix: str,
) -> set[Path]:
    referenced: set[Path] = set()
    add_error(
        errors,
        payload.get("source") == expected_source,
        f"{prefix}: source receipt differs from frozen manifest",
    )
    add_error(errors, payload.get("paper_evidence") is True, f"{prefix}: paper_evidence changed")
    add_error(
        errors,
        payload.get("runner_version") == EXPECTED_RUNNER_VERSION,
        f"{prefix}: runner version changed",
    )
    add_error(errors, isinstance(payload.get("success"), bool), f"{prefix}: success is not Boolean")
    reward = payload.get("official_reward")
    add_error(errors, isinstance(reward, (int, float)), f"{prefix}: official reward missing")
    if isinstance(reward, (int, float)) and isinstance(payload.get("success"), bool):
        add_error(
            errors,
            float(reward) in (0.0, 1.0),
            f"{prefix}: official reward is outside the frozen binary endpoint",
        )
        add_error(
            errors,
            payload["success"] == (float(reward) == 1.0),
            f"{prefix}: success and official reward disagree",
        )
    add_error(
        errors,
        payload.get("setup", {}).get("reward_remained_zero") is True,
        f"{prefix}: setup reward did not remain zero",
    )
    add_error(
        errors,
        payload.get("observation_ui", {}).get("all_clean") is True,
        f"{prefix}: accessibility overlay leaked into an observation",
    )

    for key in ("before", "before_raw"):
        receipt = payload.get(key)
        if not isinstance(receipt, dict):
            errors.append(f"{prefix}: missing {key} receipt")
            continue
        path = verify_image_receipt(receipt, attempt_dir, errors, f"{prefix}/{key}")
        if path is not None:
            referenced.add(path)

    action_executed = payload.get("action_executed") is True
    after = payload.get("after")
    if action_executed:
        add_error(errors, isinstance(after, dict), f"{prefix}: executed action lacks after image")
        add_error(
            errors,
            isinstance(payload.get("executed_action_xy"), list),
            f"{prefix}: executed action coordinate missing",
        )
        if isinstance(after, dict):
            path = verify_image_receipt(after, attempt_dir, errors, f"{prefix}/after")
            if path is not None:
                referenced.add(path)
    else:
        add_error(errors, after is None, f"{prefix}: unexecuted action has after image receipt")
        add_error(
            errors,
            payload.get("requested_action_xy") is None,
            f"{prefix}: unexecuted action has a requested coordinate",
        )
        add_error(errors, payload.get("success") is False, f"{prefix}: unexecuted action succeeded")
        add_error(
            errors,
            payload.get("scientific_failure") == "target_actor_action_parse_failure",
            f"{prefix}: unexecuted action is not a retained parse failure",
        )

    policy = payload.get("job", {}).get("policy")
    source_success = payload.get("source", {}).get("source_success") is True
    authorization = payload.get("authorization")
    if not isinstance(authorization, dict):
        errors.append(f"{prefix}: authorization receipt missing")
        return referenced
    accepted = authorization.get("accepted") is True
    calls = payload.get("model_calls")
    add_error(errors, calls in (0, 1), f"{prefix}: model call count is not 0 or 1")
    if policy == "always":
        add_error(errors, not accepted, f"{prefix}: Always unexpectedly authorized reuse")
        add_error(
            errors,
            authorization.get("reason") == "always_reground",
            f"{prefix}: Always reason changed",
        )
        add_error(errors, calls == 1, f"{prefix}: Always does not make exactly one target call")
        add_error(
            errors,
            payload.get("action_source") == "target_actor",
            f"{prefix}: Always action source changed",
        )
        add_error(
            errors,
            isinstance(payload.get("model_response"), dict),
            f"{prefix}: Always target call lacks a model response",
        )
    elif policy in {"s_only", "standard_ransac", "dafc"}:
        if not source_success:
            add_error(errors, not accepted, f"{prefix}: failed source receipt authorized reuse")
            add_error(
                errors,
                authorization.get("reason") == "source_receipt_failed",
                f"{prefix}: failed source rejection reason changed",
            )
        if accepted:
            add_error(errors, source_success, f"{prefix}: authorization lacks successful source")
            add_error(
                errors,
                isinstance(authorization.get("mapped_action_xy"), list),
                f"{prefix}: accepted route lacks mapped action",
            )
            add_error(errors, calls == 0, f"{prefix}: accepted route made a model call")
            add_error(
                errors,
                payload.get("model_response") is None,
                f"{prefix}: accepted route exposes a model response",
            )
            add_error(
                errors,
                payload.get("action_source") == "source_reuse",
                f"{prefix}: accepted route action source changed",
            )
            add_error(
                errors,
                payload.get("requested_action_xy") == authorization.get("mapped_action_xy"),
                f"{prefix}: executed request differs from authorized mapped point",
            )
        else:
            add_error(errors, calls == 1, f"{prefix}: rejected route lacks one target call")
            add_error(
                errors,
                payload.get("action_source") == "target_actor",
                f"{prefix}: rejected route action source changed",
            )
            add_error(
                errors,
                isinstance(payload.get("model_response"), dict),
                f"{prefix}: rejected route target call lacks a model response",
            )
    else:
        errors.append(f"{prefix}: unknown policy: {policy}")
    return referenced


def audit(manifest_path: Path, root: Path, allow_partial: bool) -> dict[str, Any]:
    manifest_path = Path(manifest_path).resolve()
    root = Path(root).resolve()
    errors: list[str] = []
    warnings: list[str] = []
    manifest = load_json(manifest_path)
    manifest_hash = sha256(manifest_path)
    design = manifest.get("design", {})
    planned = design.get("planned_jobs", {})

    add_error(errors, manifest_hash == EXPECTED_MANIFEST_SHA256, "manifest SHA-256 changed")
    add_error(errors, manifest.get("pass") is True, "manifest is not passing")
    add_error(errors, manifest.get("external_root") == str(root), "external result root changed")
    add_error(errors, tuple(design.get("actors", ())) == EXPECTED_ACTORS, "actor set/order changed")
    add_error(errors, tuple(design.get("policies", ())) == EXPECTED_POLICIES, "policy set/order changed")
    add_error(errors, tuple(design.get("contracts", ())) == EXPECTED_CONTRACTS, "contract set/order changed")
    add_error(
        errors,
        tuple(row.get("task_id") for row in design.get("tasks", ())) == EXPECTED_TASKS,
        "task set/order changed",
    )
    add_error(errors, canonical_sha256(planned) == manifest.get("jobs_sha256"), "canonical job hash changed")
    add_error(errors, manifest.get("jobs_sha256") == EXPECTED_JOBS_SHA256, "frozen job SHA-256 changed")
    add_error(
        errors,
        canonical_sha256(design.get("tasks")) == manifest.get("sources_sha256"),
        "canonical source hash changed",
    )
    expected_sources = {
        row.get("task_id"): row.get("source") for row in design.get("tasks", ())
    }

    expected: dict[str, dict[str, dict[str, Any]]] = {}
    all_expected_ids: set[str] = set()
    for actor in EXPECTED_ACTORS:
        actor_jobs = planned.get(actor, [])
        expected[actor] = {row.get("job_id"): row for row in actor_jobs}
        add_error(errors, len(actor_jobs) == 560, f"{actor}: planned job count is not 560")
        add_error(errors, len(expected[actor]) == 560, f"{actor}: planned job IDs are not unique")
        all_expected_ids.update(expected[actor])
    add_error(errors, len(all_expected_ids) == 1120, "planned job IDs are not globally unique")

    attempts_by_job: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    result_paths: dict[str, set[Path]] = {actor: set() for actor in EXPECTED_ACTORS}
    result_hashes: dict[str, str] = {}
    referenced_images: set[Path] = set()
    expected_image_dirs: set[Path] = set()
    results_root = root / "results"
    if results_root.is_dir():
        for path in sorted(results_root.rglob("*")):
            if not path.is_file():
                continue
            if path.suffix != ".json":
                errors.append(f"unexpected non-JSON result artifact: {path}")
            elif path.parent.name not in EXPECTED_ACTORS or path.parent.parent != results_root:
                errors.append(f"unexpected result location: {path}")
    elif not allow_partial:
        errors.append(f"results directory missing: {results_root}")

    for actor in EXPECTED_ACTORS:
        actor_dir = results_root / actor
        for path in sorted(actor_dir.glob("*.json")) if actor_dir.is_dir() else []:
            prefix = f"{actor}/{path.name}"
            resolved_path = path.resolve()
            if resolved_path.parent != actor_dir.resolve():
                errors.append(f"{prefix}: result path escapes exact actor directory")
                continue
            result_paths[actor].add(resolved_path)
            result_hashes[resolved_path.relative_to(root).as_posix()] = sha256(path)
            try:
                payload = load_json(path)
            except Exception as caught:
                errors.append(f"{prefix}: unreadable result JSON: {caught}")
                continue
            job = payload.get("job")
            if not isinstance(job, dict):
                errors.append(f"{prefix}: missing job object")
                continue
            job_id = job.get("job_id")
            expected_job = expected[actor].get(job_id)
            if expected_job is None:
                errors.append(f"{prefix}: job is outside frozen plan: {job_id}")
                continue
            add_error(errors, job == expected_job, f"{prefix}: job payload differs from manifest")
            attempt = payload.get("attempt")
            add_error(errors, attempt in (0, 1), f"{prefix}: attempt index is outside 0/1")
            if attempt not in (0, 1):
                continue
            expected_name = f"{job_id}--a{attempt:02d}.json"
            add_error(errors, path.name == expected_name, f"{prefix}: result filename disagrees with payload")
            add_error(
                errors,
                payload.get("manifest_sha256") == manifest_hash,
                f"{prefix}: manifest binding changed",
            )
            add_error(
                errors,
                payload.get("jobs_sha256") == manifest.get("jobs_sha256"),
                f"{prefix}: job-plan binding changed",
            )
            add_error(
                errors,
                payload.get("paper_evidence") is True,
                f"{prefix}: paper_evidence changed",
            )
            add_error(
                errors,
                payload.get("runner_version") == EXPECTED_RUNNER_VERSION,
                f"{prefix}: runner version changed",
            )
            add_error(
                errors,
                expected_execution_manifest_path(payload.get("manifest")),
                f"{prefix}: manifest path binding changed",
            )
            key = (actor, job_id)
            if any(row.get("attempt") == attempt for row in attempts_by_job[key]):
                errors.append(f"{prefix}: duplicate attempt index")
            attempts_by_job[key].append(payload)

            outcome = payload.get("outcome_bearing")
            if outcome is True:
                add_error(
                    errors,
                    payload.get("retry_permitted") is False,
                    f"{prefix}: outcome-bearing attempt permits retry",
                )
                if payload.get("status") == "complete":
                    attempt_dir = (root / "images" / actor / f"{job_id}--a{attempt:02d}").resolve()
                    expected_image_dirs.add(attempt_dir)
                    referenced_images.update(
                        validate_complete_payload(
                            payload,
                            expected_sources.get(job.get("task")),
                            attempt_dir,
                            errors,
                            prefix,
                        )
                    )
                else:
                    add_error(
                        errors,
                        payload.get("status") == "outcome_failure",
                        f"{prefix}: invalid outcome-bearing status",
                    )
                    add_error(errors, payload.get("success") is False, f"{prefix}: outcome failure succeeded")
                    add_error(
                        errors,
                        payload.get("retry_permitted") is False,
                        f"{prefix}: outcome failure permits retry",
                    )
            elif outcome is False:
                add_error(
                    errors,
                    payload.get("status") == "infrastructure_failure",
                    f"{prefix}: non-outcome attempt status changed",
                )
                add_error(errors, payload.get("success") is None, f"{prefix}: infrastructure attempt has outcome")
                add_error(errors, payload.get("model_calls") == 0, f"{prefix}: infrastructure attempt made a model call")
                add_error(
                    errors,
                    payload.get("authorization_started") is False,
                    f"{prefix}: infrastructure attempt began authorization",
                )
                add_error(
                    errors,
                    payload.get("target_action_started") is False,
                    f"{prefix}: infrastructure attempt began a target action",
                )
            else:
                errors.append(f"{prefix}: outcome_bearing is not Boolean")

    ledger_hashes: dict[str, str] = {}
    ledger_paths: dict[str, set[Path]] = {actor: set() for actor in EXPECTED_ACTORS}
    ledger_root = root / "ledger"
    if ledger_root.is_dir():
        unexpected_ledgers = [
            path for path in ledger_root.iterdir()
            if path.is_file() and path.name not in {f"{actor}.jsonl" for actor in EXPECTED_ACTORS}
        ]
        errors.extend(f"unexpected ledger file: {path}" for path in unexpected_ledgers)
    for actor in EXPECTED_ACTORS:
        path = ledger_root / f"{actor}.jsonl"
        if not path.is_file():
            if result_paths[actor] or not allow_partial:
                errors.append(f"{actor}: ledger is missing")
            continue
        resolved_ledger = path.resolve()
        if resolved_ledger.parent != ledger_root.resolve():
            errors.append(f"{actor}: ledger path escapes exact ledger directory")
            continue
        ledger_hashes[resolved_ledger.relative_to(root).as_posix()] = sha256(path)
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            prefix = f"{path.name}:{line_number}"
            try:
                row = json.loads(line)
            except Exception as caught:
                errors.append(f"{prefix}: invalid JSONL row: {caught}")
                continue
            try:
                attempt_path = Path(row["attempt_path"]).resolve()
            except Exception as caught:
                errors.append(f"{prefix}: invalid attempt path: {caught}")
                continue
            if attempt_path in ledger_paths[actor]:
                errors.append(f"{prefix}: duplicate ledger attempt path")
            ledger_paths[actor].add(attempt_path)
            if not within(attempt_path, results_root / actor) or attempt_path.parent != (results_root / actor).resolve():
                errors.append(f"{prefix}: ledger attempt escapes actor result directory")
                continue
            if not attempt_path.is_file():
                errors.append(f"{prefix}: ledger result is missing: {attempt_path}")
                continue
            try:
                payload = load_json(attempt_path)
            except Exception as caught:
                errors.append(f"{prefix}: ledger result is unreadable: {caught}")
                continue
            expected_row = {
                "attempt_path": str(attempt_path),
                "attempt_sha256": sha256(attempt_path),
                "job_id": payload.get("job", {}).get("job_id"),
                "attempt": payload.get("attempt"),
                "status": payload.get("status"),
                "outcome_bearing": payload.get("outcome_bearing"),
                "retry_permitted": payload.get("retry_permitted"),
            }
            add_error(errors, row == expected_row, f"{prefix}: ledger row differs from immutable result")
        add_error(
            errors,
            ledger_paths[actor] == result_paths[actor],
            f"{actor}: ledger/result attempt sets differ",
        )

    completed_jobs = 0
    pending_retry_jobs = 0
    untouched_jobs = 0
    outcome_rows: list[dict[str, Any]] = []
    for actor in EXPECTED_ACTORS:
        for job_id in expected[actor]:
            attempts = sorted(attempts_by_job.get((actor, job_id), []), key=lambda row: row["attempt"])
            local_errors = sequence_errors(attempts, require_complete=not allow_partial)
            errors.extend(f"{actor}/{job_id}: {message}" for message in local_errors)
            outcomes = [row for row in attempts if row.get("outcome_bearing") is True]
            if len(outcomes) == 1:
                completed_jobs += 1
                outcome_rows.append(outcomes[0])
            elif attempts:
                pending_retry_jobs += 1
            else:
                untouched_jobs += 1

    image_hashes: dict[str, str] = {}
    observed_images: set[Path] = set()
    images_root = root / "images"
    if images_root.is_dir():
        for path in sorted(images_root.rglob("*")):
            if not path.is_file():
                continue
            resolved = path.resolve()
            try:
                relative = resolved.relative_to(root).as_posix()
                image_relative = resolved.relative_to(images_root)
            except ValueError:
                errors.append(f"image artifact escapes experiment root: {path}")
                continue
            if path.suffix.lower() != ".png":
                errors.append(f"unexpected non-PNG image artifact: {path}")
                continue
            if len(image_relative.parts) != 3:
                errors.append(f"unexpected image layout: {path}")
                continue
            actor, attempt_name, filename = image_relative.parts
            attempt_dir = (images_root / actor / attempt_name).resolve()
            if actor not in EXPECTED_ACTORS:
                errors.append(f"unexpected image actor directory: {path}")
            if attempt_dir not in expected_image_dirs and not (
                results_root / actor / f"{attempt_name}.json"
            ).is_file():
                errors.append(f"image directory has no matching result attempt: {path}")
            if filename not in {"before.png", "before_raw.png", "after.png"}:
                errors.append(f"unexpected image filename: {path}")
            observed_images.add(resolved)
            image_hashes[relative] = sha256(path)
    missing_references = [path for path in referenced_images if not path.is_file()]
    errors.extend(f"referenced image disappeared: {path}" for path in missing_references)
    unreferenced_images = observed_images - referenced_images

    actor_counts = Counter(row.get("job", {}).get("actor") for row in outcome_rows)
    actor_policy_counts = Counter(
        (row.get("job", {}).get("actor"), row.get("job", {}).get("policy"))
        for row in outcome_rows
    )
    if not allow_partial:
        add_error(errors, completed_jobs == 1120, "final run lacks exactly 1120 outcome rows")
        add_error(errors, pending_retry_jobs == 0, "final run has a pending retry")
        add_error(errors, untouched_jobs == 0, "final run has untouched jobs")
        for actor in EXPECTED_ACTORS:
            add_error(errors, actor_counts[actor] == 560, f"{actor}: final outcome count is not 560")
            for policy in EXPECTED_POLICIES:
                add_error(
                    errors,
                    actor_policy_counts[(actor, policy)] == 140,
                    f"{actor}/{policy}: final outcome count is not 140",
                )
    elif pending_retry_jobs:
        warnings.append(f"{pending_retry_jobs} job(s) are waiting for the single permitted retry")

    snapshot = {
        "result_files": result_hashes,
        "ledger_files": ledger_hashes,
        "image_files": image_hashes,
    }
    analysis_attempt_bindings = {
        str((root / relative).resolve()): digest
        for relative, digest in result_hashes.items()
    }
    criteria = {
        "all_observed_integrity_checks_pass": not errors,
        "performance_not_used_as_integrity_gate": True,
        "zero_model_or_environment_calls_during_audit": True,
        "complete_1120_job_matrix": completed_jobs == 1120 and not pending_retry_jobs and not untouched_jobs,
    }
    if allow_partial:
        criteria["complete_1120_job_matrix"] = False
    result = {
        "title": "C589 Native-Android Atomic Completion Integrity Audit",
        "mode": "partial_read_only" if allow_partial else "complete_read_only",
        "paper_evidence": not allow_partial,
        "bindings": {
            "manifest": str(manifest_path),
            "manifest_sha256": manifest_hash,
            "jobs_sha256": manifest.get("jobs_sha256"),
            "root": str(root),
            "audit_script": str(Path(__file__).resolve()),
            "audit_script_sha256": sha256(Path(__file__).resolve()),
            "analysis_attempt_snapshot_sha256": canonical_sha256(
                analysis_attempt_bindings
            ),
            "execution_snapshot_sha256": canonical_sha256(snapshot),
        },
        "counts": {
            "planned_jobs": 1120,
            "completed_jobs": completed_jobs,
            "pending_retry_jobs": pending_retry_jobs,
            "untouched_jobs": untouched_jobs,
            "result_attempt_files": len(result_hashes),
            "ledger_rows": sum(len(rows) for rows in ledger_paths.values()),
            "image_files": len(image_hashes),
            "unreferenced_failure_image_files": len(unreferenced_images),
            "actor_outcomes": dict(sorted(actor_counts.items())),
            "actor_policy_outcomes": {
                f"{actor}/{policy}": actor_policy_counts[(actor, policy)]
                for actor in EXPECTED_ACTORS
                for policy in EXPECTED_POLICIES
            },
        },
        "criteria": criteria,
        "warnings": warnings,
        "errors": errors,
        "pass": not errors and (allow_partial or criteria["complete_1120_job_matrix"]),
    }
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="audit the immutable observed prefix without requiring all 1,120 jobs",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="write a new final audit JSON; allowed only for a complete audit",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.output is not None and args.allow_partial:
        raise RuntimeError("partial audits cannot write the final evidence artifact")
    if args.output is not None:
        output = args.output if args.output.is_absolute() else PROJECT / args.output
        output = output.resolve()
        if output != FINAL_OUTPUT.resolve():
            raise RuntimeError(f"final audit output path changed: {output}")
    if args.output is not None and not committed_and_clean(Path(__file__).resolve()):
        raise RuntimeError("final audit output requires committed code and a clean repository")
    result = audit(args.manifest, args.root, args.allow_partial)
    if args.output is not None:
        if not result["pass"]:
            print(json.dumps(result, indent=2, sort_keys=True))
            return 1
        if output.exists() or output.is_symlink():
            raise RuntimeError(f"refusing to overwrite audit output: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
