"""Read-only component-baseline audit for frozen TransitionGuard V5 logs.

The analysis is bound by
reports/transitionguard_v6_postcheck_baseline_preregistration.md. It validates
every ledger and attempt hash before reading the logged component decisions.
It never imports the runtime, recomputes a threshold, or mutates a formal root.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

AUDIT_VERSION = "transitionguard-v6-postcheck-component-baselines-v1"
BOOTSTRAP_SEED = 20260719
BOOTSTRAP_RESAMPLES = 10000
DETECTORS = ("change_only", "absolute_template", "directional_effect", "transition_contract")
SEMANTIC_CLASSES = ("advanced", "nonadvancing", "harmful")
EXPECTED_LEDGER_HASHES = {
    "natural/qwen25vl": "d887b636c17106edbd170588c01854a8492f137c32c75bf3da70101f2fda7ae3",
    "natural/uitars": "b6425abe901c1c2b42cfa403acf32a09165e0d84cfeec63c93f0ff39ec6c2401",
    "drop/qwen25vl": "03d16af114ba1f2d07cbfc44ecd78501f20c844f509b884cde06e6dfc61e9360",
    "drop/uitars": "e956883d4e89e1146e6a3a948e82c96a2cf20de6ac44ba9d272b74a3489e7225",
    "adjacent/qwen25vl": "a69274b6128072683453d8c2d97bafdccbe461fc4e99e12d85ca7adc7017f06d",
    "adjacent/uitars": "be872b0b8ad66de3bb68369d01541f0da39e48e77e06eec3af36db51b55a0ad8",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _required_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"required Boolean field missing or invalid: {field}")
    return value


def detector_decisions(post: dict[str, Any]) -> dict[str, bool]:
    """Extract the four preregistered decisions from one logged post-check."""
    if not isinstance(post, dict):
        raise ValueError("post_verification must be an object")
    absolute = post.get("absolute_postcondition")
    relative = post.get("relative_effect")
    if not isinstance(absolute, dict) or not isinstance(relative, dict):
        raise ValueError("post-check component objects are missing")
    checks = relative.get("checks")
    if not isinstance(checks, dict):
        raise ValueError("relative_effect.checks is missing")
    decisions = {
        "change_only": _required_bool(checks.get("target_change"), "relative_effect.checks.target_change"),
        "absolute_template": _required_bool(absolute.get("accepted"), "absolute_postcondition.accepted"),
        "directional_effect": _required_bool(relative.get("accepted"), "relative_effect.accepted"),
    }
    decisions["transition_contract"] = bool(decisions["absolute_template"] or decisions["directional_effect"])
    logged = _required_bool(post.get("post_verified"), "post_verified")
    if logged != decisions["transition_contract"]:
        raise ValueError("logged post_verified disagrees with absolute OR directional effect")
    return decisions


def read_completed_full_attempts(
    *, root: Path, ledger_relative: Path, expected_ledger_sha256: str
) -> tuple[list[tuple[dict[str, Any], dict[str, Any], Path]], dict[str, Any]]:
    """Load completed Full attempts after ledger and artifact validation."""
    ledger = root / ledger_relative
    observed_ledger_sha256 = sha256(ledger)
    if observed_ledger_sha256 != expected_ledger_sha256:
        raise ValueError(f"ledger hash mismatch for {ledger}: {observed_ledger_sha256}")
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines() if line.strip()]
    selected = [row for row in rows if row.get("status") == "completed" and row.get("policy") == "full"]
    logical_ids = [str(row.get("logical_id")) for row in selected]
    if len(logical_ids) != len(set(logical_ids)):
        raise ValueError(f"duplicate completed Full logical IDs in {ledger}")
    attempts: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    for row in selected:
        artifact = root / str(row.get("artifact"))
        observed_artifact_sha256 = sha256(artifact)
        if observed_artifact_sha256 != row.get("artifact_sha256"):
            raise ValueError(f"artifact hash mismatch: {artifact}")
        document = load_object(artifact)
        if document.get("policy") != "full":
            raise ValueError(f"selected artifact is not Full: {artifact}")
        attempts.append((row, document, artifact))
    receipt = {
        "completed_full_attempts": len(attempts),
        "ledger": ledger_relative.as_posix(),
        "ledger_rows": len(rows),
        "ledger_sha256": observed_ledger_sha256,
        "validated_artifacts": len(attempts),
    }
    return (attempts, receipt)


def _unit(
    *, actor: str, row: dict[str, Any], step: dict[str, Any], artifact: Path, semantic_required: bool
) -> dict[str, Any]:
    post = step.get("post_verification")
    if not isinstance(post, dict):
        raise ValueError(f"eligible step has no post_verification: {artifact}")
    semantic_class = None
    if semantic_required:
        semantic = step.get("scoring_only_semantic_transition")
        if not isinstance(semantic, dict):
            raise ValueError(f"natural step lacks independent semantics: {artifact}")
        if semantic.get("independent_of_visual_guard") is not True:
            raise ValueError(f"semantic label is not guard-independent: {artifact}")
        semantic_class = semantic.get("classification")
        if semantic_class not in SEMANTIC_CLASSES:
            raise ValueError(f"unexpected semantic class in {artifact}: {semantic_class}")
    step_index = step.get("step")
    if not isinstance(step_index, int):
        raise ValueError(f"eligible step index is invalid: {artifact}")
    return {
        "actor": actor,
        "contract": row.get("contract"),
        "decisions": detector_decisions(post),
        "logical_id": row.get("logical_id"),
        "semantic_class": semantic_class,
        "source_cluster_id": row.get("source_cluster_id"),
        "step": step_index,
        "task": row.get("task"),
    }


def natural_units(
    actor: str, attempts: Iterable[tuple[dict[str, Any], dict[str, Any], Path]]
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for row, document, artifact in attempts:
        for step in document.get("steps", []):
            if step.get("action_source") != "transition_guard":
                continue
            units.append(_unit(actor=actor, row=row, step=step, artifact=artifact, semantic_required=True))
    return units


def _step_by_index(document: dict[str, Any], index: int, artifact: Path) -> dict[str, Any]:
    matches = [step for step in document.get("steps", []) if step.get("step") == index]
    if len(matches) != 1:
        raise ValueError(f"expected one injected step {index}: {artifact}")
    step = matches[0]
    if step.get("action_source") != "transition_guard":
        raise ValueError(f"injected step is not a guarded action: {artifact}")
    return step


def drop_units(
    actor: str, attempts: Iterable[tuple[dict[str, Any], dict[str, Any], Path]]
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for row, document, artifact in attempts:
        instrumentation = document.get("disturbance_instrumentation")
        dropped = instrumentation.get("dropped_steps") if isinstance(instrumentation, dict) else None
        if not isinstance(dropped, list) or len(dropped) != 1 or (not isinstance(dropped[0], int)):
            raise ValueError(f"expected exactly one dropped step: {artifact}")
        units.append(
            _unit(
                actor=actor,
                row=row,
                step=_step_by_index(document, dropped[0], artifact),
                artifact=artifact,
                semantic_required=False,
            )
        )
    return units


def adjacent_units(
    actor: str, attempts: Iterable[tuple[dict[str, Any], dict[str, Any], Path]]
) -> list[dict[str, Any]]:
    units: list[dict[str, Any]] = []
    for row, document, artifact in attempts:
        instrumentation = document.get("adjacent_substitution_instrumentation")
        injection = instrumentation.get("injection") if isinstance(instrumentation, dict) else None
        index = injection.get("episode_step") if isinstance(injection, dict) else None
        if not isinstance(index, int):
            raise ValueError(f"adjacent injection step is missing: {artifact}")
        units.append(
            _unit(
                actor=actor,
                row=row,
                step=_step_by_index(document, index, artifact),
                artifact=artifact,
                semantic_required=False,
            )
        )
    return units


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def summarize_natural(units: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "evaluated_steps": len(units),
        "semantic_totals": dict(sorted(Counter((unit["semantic_class"] for unit in units)).items())),
    }
    detector_rows: dict[str, Any] = {}
    for detector in DETECTORS:
        detector_rows[detector] = _detector_summary(units, detector)
    relative_only = [
        unit
        for unit in units
        if unit["decisions"]["transition_contract"] and (not unit["decisions"]["absolute_template"])
    ]
    summary["detectors"] = detector_rows
    summary["relative_only_acceptances"] = {
        "count": len(relative_only),
        "semantics": {
            name: sum((unit["semantic_class"] == name for unit in relative_only)) for name in SEMANTIC_CLASSES
        },
    }
    return summary


def disagreement(units: list[dict[str, Any]], baseline: str) -> dict[str, int]:
    counts = Counter()
    for unit in units:
        contract = unit["decisions"]["transition_contract"]
        comparison = unit["decisions"][baseline]
        if contract and comparison:
            counts["both_accept"] += 1
        elif contract:
            counts["contract_only"] += 1
        elif comparison:
            counts["baseline_only"] += 1
        else:
            counts["both_reject"] += 1
    return {key: counts[key] for key in ("both_accept", "contract_only", "baseline_only", "both_reject")}


def summarize_fault(units: list[dict[str, Any]]) -> dict[str, Any]:
    detector_rows = {}
    for detector in DETECTORS:
        accepted = sum((unit["decisions"][detector] for unit in units))
        rejected = len(units) - accepted
        detector_rows[detector] = {
            "accepted": accepted,
            "acceptance_rate": _rate(accepted, len(units)),
            "rejected": rejected,
            "rejection_rate": _rate(rejected, len(units)),
        }
    return {
        "detectors": detector_rows,
        "evaluated_steps": len(units),
        "paired_disagreement": {
            baseline: disagreement(units, baseline) for baseline in ("change_only", "absolute_template")
        },
    }


def _advanced_difference(units: list[dict[str, Any]]) -> float:
    advanced = [unit for unit in units if unit["semantic_class"] == "advanced"]
    if not advanced:
        raise ValueError("bootstrap sample contains no advanced action")
    contract = sum((unit["decisions"]["transition_contract"] for unit in advanced))
    absolute = sum((unit["decisions"]["absolute_template"] for unit in advanced))
    return (contract - absolute) / len(advanced)


def _percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def cluster_bootstrap(units: list[dict[str, Any]], *, stratify_actor: bool) -> dict[str, Any]:
    actors = sorted({unit["actor"] for unit in units})
    strata = actors if stratify_actor else ["all"]
    by_stratum: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for stratum in strata:
        eligible = units if stratum == "all" else [u for u in units if u["actor"] == stratum]
        clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for unit in eligible:
            clusters[str(unit["source_cluster_id"])].append(unit)
        by_stratum[stratum] = dict(clusters)
    rng = random.Random(BOOTSTRAP_SEED)
    estimates: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        sampled: list[dict[str, Any]] = []
        for stratum in strata:
            clusters = by_stratum[stratum]
            cluster_ids = sorted(clusters)
            for _ in cluster_ids:
                sampled.extend(clusters[rng.choice(cluster_ids)])
        estimates.append(_advanced_difference(sampled))
    return {
        "cluster_count": sum((len(clusters) for clusters in by_stratum.values())),
        "confidence_interval_95": [_percentile(estimates, 0.025), _percentile(estimates, 0.975)],
        "estimate": _advanced_difference(units),
        "resamples": BOOTSTRAP_RESAMPLES,
        "seed": BOOTSTRAP_SEED,
        "stratified_by_actor": stratify_actor,
    }


def grouped_natural(units: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = sorted({str(unit[field]) for unit in units})
    return {
        value: summarize_natural([unit for unit in units if str(unit[field]) == value]) for value in values
    }


def grouped_fault(units: list[dict[str, Any]], field: str) -> dict[str, Any]:
    values = sorted({str(unit[field]) for unit in units})
    return {value: summarize_fault([unit for unit in units if str(unit[field]) == value]) for value in values}


def interpretation_gate(report: dict[str, Any]) -> dict[str, Any]:
    natural = report["natural"]["pooled"]
    adjacent = report["faults"]["adjacent"]["pooled"]
    drop = report["faults"]["drop"]["pooled"]
    bootstrap = report["natural"]["advanced_contract_minus_absolute_bootstrap"]["pooled"]
    natural_difference = (
        natural["detectors"]["transition_contract"]["accepted_advanced_rate"]
        - natural["detectors"]["absolute_template"]["accepted_advanced_rate"]
    )
    adjacent_contract_rejection = adjacent["detectors"]["transition_contract"]["rejection_rate"]
    adjacent_change_rejection = adjacent["detectors"]["change_only"]["rejection_rate"]
    drop_contract_rejection = drop["detectors"]["transition_contract"]["rejection_rate"]
    checks = {
        "advanced_acceptance_gain_at_least_5pp": natural_difference >= 0.05,
        "advanced_gain_ci_lower_above_zero": bootstrap["confidence_interval_95"][0] > 0.0,
        "adjacent_contract_rejection_at_least_90pct": adjacent_contract_rejection >= 0.9,
        "adjacent_rejection_gain_over_change_at_least_10pp": adjacent_contract_rejection
        - adjacent_change_rejection
        >= 0.1,
        "all_dropped_actions_rejected": drop_contract_rejection == 1.0,
    }
    return {
        "checks": checks,
        "observed": {
            "advanced_acceptance_gain": natural_difference,
            "advanced_gain_ci_lower": bootstrap["confidence_interval_95"][0],
            "adjacent_contract_rejection": adjacent_contract_rejection,
            "adjacent_rejection_gain_over_change": adjacent_contract_rejection - adjacent_change_rejection,
            "drop_contract_rejection": drop_contract_rejection,
        },
        "passed": all(checks.values()),
    }


def audit(formal_root: Path, adjacent_root: Path) -> dict[str, Any]:
    formal_root = formal_root.resolve()
    adjacent_root = adjacent_root.resolve()
    inputs = {
        "natural/qwen25vl": (formal_root, Path("target/qwen25vl/ledger.jsonl")),
        "natural/uitars": (formal_root, Path("target/uitars/ledger.jsonl")),
        "drop/qwen25vl": (formal_root, Path("stress/dropped_action_receipt_repair/qwen25vl/ledger.jsonl")),
        "drop/uitars": (formal_root, Path("stress/dropped_action_receipt_repair/uitars/ledger.jsonl")),
        "adjacent/qwen25vl": (adjacent_root, Path("stress/adjacent_substitution/qwen25vl/ledger.jsonl")),
        "adjacent/uitars": (adjacent_root, Path("stress/adjacent_substitution/uitars/ledger.jsonl")),
    }
    loaded: dict[str, list[tuple[dict[str, Any], dict[str, Any], Path]]] = {}
    receipts: dict[str, Any] = {}
    for name, (root, ledger) in inputs.items():
        loaded[name], receipts[name] = read_completed_full_attempts(
            root=root, ledger_relative=ledger, expected_ledger_sha256=EXPECTED_LEDGER_HASHES[name]
        )
    natural: list[dict[str, Any]] = []
    drops: list[dict[str, Any]] = []
    adjacent: list[dict[str, Any]] = []
    for actor in ("qwen25vl", "uitars"):
        natural.extend(natural_units(actor, loaded[f"natural/{actor}"]))
        drops.extend(drop_units(actor, loaded[f"drop/{actor}"]))
        adjacent.extend(adjacent_units(actor, loaded[f"adjacent/{actor}"]))
    report: dict[str, Any] = {
        "audit_version": AUDIT_VERSION,
        "detector_definitions": {
            "absolute_template": "absolute_postcondition.accepted",
            "change_only": "relative_effect.checks.target_change",
            "directional_effect": "relative_effect.accepted",
            "transition_contract": "absolute_template OR directional_effect",
        },
        "input_receipts": receipts,
        "limitations": [
            "secondary post-result mechanism analysis",
            "observed guarded actions only; no counterfactual episode success",
            "component baselines are not reproductions of published systems",
        ],
        "natural": {
            "advanced_contract_minus_absolute_bootstrap": {
                "pooled": cluster_bootstrap(natural, stratify_actor=True),
                "qwen25vl": cluster_bootstrap(
                    [unit for unit in natural if unit["actor"] == "qwen25vl"], stratify_actor=False
                ),
                "uitars": cluster_bootstrap(
                    [unit for unit in natural if unit["actor"] == "uitars"], stratify_actor=False
                ),
            },
            "by_actor": grouped_natural(natural, "actor"),
            "by_contract": grouped_natural(natural, "contract"),
            "by_task": grouped_natural(natural, "task"),
            "pooled": summarize_natural(natural),
        },
        "preregistration": "reports/transitionguard_v6_postcheck_baseline_preregistration.md",
        "scope": "read-only logged component comparison over frozen V5 evidence",
        "faults": {
            "adjacent": {"by_actor": grouped_fault(adjacent, "actor"), "pooled": summarize_fault(adjacent)},
            "drop": {"by_actor": grouped_fault(drops, "actor"), "pooled": summarize_fault(drops)},
        },
    }
    report["expectations"] = {
        "adjacent_full_attempts_are_48": len(adjacent) == 48,
        "drop_full_attempts_are_48": len(drops) == 48,
        "ledger_hashes_match_preregistration": all(
            (receipts[name]["ledger_sha256"] == expected for name, expected in EXPECTED_LEDGER_HASHES.items())
        ),
        "natural_has_both_actors": set((unit["actor"] for unit in natural)) == {"qwen25vl", "uitars"},
        "natural_units_are_nonempty": bool(natural),
    }
    report["expectations_passed"] = all(report["expectations"].values())
    report["interpretation_gate"] = interpretation_gate(report)
    return report


def main() -> int:
    from dafc.evaluation.postcheck_report import latex_numbers, latex_table, markdown_report

    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, required=True)
    parser.add_argument("--adjacent-root", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--output-tex", type=Path)
    parser.add_argument("--output-numbers-tex", type=Path)
    args = parser.parse_args()
    report = audit(args.formal_root, args.adjacent_root)
    serialized = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized, encoding="utf-8")
    else:
        print(serialized, end="")
    if args.output_md:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text(markdown_report(report), encoding="utf-8")
    if args.output_tex:
        args.output_tex.parent.mkdir(parents=True, exist_ok=True)
        args.output_tex.write_text(latex_table(report), encoding="utf-8")
    if args.output_numbers_tex:
        args.output_numbers_tex.parent.mkdir(parents=True, exist_ok=True)
        args.output_numbers_tex.write_text(latex_numbers(report), encoding="utf-8")
    return int(not report["expectations_passed"])


if __name__ == "__main__":
    raise SystemExit(main())


def _detector_summary(units, detector):
    accepted = [unit for unit in units if unit["decisions"][detector]]
    accepted_semantics = Counter((unit["semantic_class"] for unit in accepted))
    rejected_semantics = Counter(
        (unit["semantic_class"] for unit in units if not unit["decisions"][detector])
    )
    advanced_total = sum((unit["semantic_class"] == "advanced" for unit in units))
    adverse_total = len(units) - advanced_total
    accepted_advanced = accepted_semantics["advanced"]
    accepted_adverse = accepted_semantics["nonadvancing"] + accepted_semantics["harmful"]
    return {
        "accepted": len(accepted),
        "accepted_advanced": accepted_advanced,
        "accepted_advanced_rate": _rate(accepted_advanced, advanced_total),
        "accepted_adverse": accepted_adverse,
        "accepted_adverse_rate": _rate(accepted_adverse, adverse_total),
        "accepted_semantics": {name: accepted_semantics[name] for name in SEMANTIC_CLASSES},
        "acceptance_rate": _rate(len(accepted), len(units)),
        "rejected": len(units) - len(accepted),
        "rejected_semantics": {name: rejected_semantics[name] for name in SEMANTIC_CLASSES},
    }
