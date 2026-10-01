#!/usr/bin/env python3
"""Run the frozen Android V2 analysis with a representation-only repair."""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import argparse
import copy
import json
import math
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator


from dafc.paths import PROJECT
FROZEN_PROJECT = Path(__import__("os").environ.get("DAFC_FROZEN_V2_ROOT", str(PROJECT / "resources/frozen/android-v2")))
FROZEN_SCRIPTS = FROZEN_PROJECT / "scripts"
FROZEN_HEAD = "ca007e64273c6902aa2b9f5f83d7acf1f07e211e"
FROZEN_ANALYZER_SHA256 = (
    "2340f2b3e5a69756b3657b016fe0ab63293321702a9a526e23c8ff61d8836253"
)
FROZEN_MANIFEST_SHA256 = (
    "4077d203fb47489659323c7907200e07018647ee8bf9bf725d7337f24ebe9951"
)
FROZEN_LIVE_RUNNER_SHA256 = (
    "55f0f82b1c67f84b714e6b77ce92e1191548950a7c0bcd3a43ef47a81ee36539"
)
FROZEN_C488_SHA256 = (
    "b4beaedcd14a7e2889466245cb3ad2a6b8154ca71f70153ca3714fe9dcc9e597"
)
REPAIR_VERSION = (
    "android-recovered-source-followup-v2-integral-coordinate-loader-repair-v1"
)
QUALIFICATION_VERSION = (
    "c576-android-recovered-integral-coordinate-repair-qualification-v1"
)
ANALYSIS_VERSION = (
    "android-recovered-source-conditional-descriptive-v2-"
    "full-provenance-integral-coordinate-loader-repaired"
)
REPAIR_PATH = (
    PROJECT
    / "formal"
    / "transitionguard_android_recovered_source_followup_v2_"
    "integral_coordinate_repair.json"
)
QUALIFICATION_PATH = (
    PROJECT
    / "formal"
    / "transitionguard_android_recovered_source_followup_v2_"
    "integral_coordinate_qualification.json"
)
FROZEN_MANIFEST = (
    FROZEN_PROJECT
    / "formal"
    / "transitionguard_android_recovered_source_followup_v2_manifest.json"
)
FROZEN_ANALYZER = (
    FROZEN_SCRIPTS / "analyze_android_recovered_source_followup_v2.py"
)
FROZEN_LIVE_RUNNER = FROZEN_SCRIPTS / "live_androidworld_transition_guard.py"
FROZEN_C488 = FROZEN_SCRIPTS / "c488_android_live_runner_integrity_audit.py"
ANALYSIS_OUTPUT = (
    Path("data/formal_transitionguard_android_recovered_v2")
    / "androidworld"
    / "analysis"
    / "recovered_source_followup_v2_analysis.json"
)

sys.path.insert(0, str(FROZEN_SCRIPTS))
frozen_analysis = LazyModule('analyze_android_recovered_source_followup_v2')
frozen_followup = LazyModule('formal_android_recovered_source_followup_v2')
import dafc.evidence as frozen_formal


Decorator = Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]


def git(cwd: Path, *args: str) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def strict_json_object(path: Path) -> dict[str, Any]:
    def unique_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"duplicate JSON key: {path}")
            result[key] = value
        return result

    payload = json.loads(
        path.read_text(encoding="utf-8"),
        object_pairs_hook=unique_pairs,
        parse_constant=lambda value: (_ for _ in ()).throw(
            RuntimeError(f"nonfinite JSON number in {path}: {value}")
        ),
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return payload


def validate_frozen_checkout() -> None:
    if (
        not FROZEN_PROJECT.is_dir()
        or FROZEN_PROJECT.is_symlink()
        or git(FROZEN_PROJECT, "rev-parse", "HEAD") != FROZEN_HEAD
        or git(
            FROZEN_PROJECT,
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
        )
        != ""
        or frozen_formal.sha256(FROZEN_ANALYZER)
        != FROZEN_ANALYZER_SHA256
        or frozen_formal.sha256(FROZEN_MANIFEST)
        != FROZEN_MANIFEST_SHA256
        or frozen_formal.sha256(FROZEN_LIVE_RUNNER)
        != FROZEN_LIVE_RUNNER_SHA256
        or frozen_formal.sha256(FROZEN_C488) != FROZEN_C488_SHA256
    ):
        raise RuntimeError("frozen Android V2 checkout binding changed")


from dafc.coordinates import (
    _empty_actor_audit as _empty_actor_audit,
    normalize_integral_action_coordinates as normalize_integral_action_coordinates,
)




@contextmanager
def use_integral_coordinate_repair(
    decorator: Decorator | None = None,
) -> Iterator[dict[str, Any]]:
    """Install the narrow validator adapter and always restore frozen globals."""

    original_membership = frozen_analysis.validate_result_membership
    original_complete = frozen_analysis.analyze_complete
    state: dict[str, Any] = {}

    def repaired_membership(
        experiment,
        plan,
        rows_by_actor,
        completed_payloads,
        orphan_record_ids,
    ):
        if "coordinate_audit" in state:
            raise RuntimeError("coordinate repair was invoked more than once")
        normalized, audit = normalize_integral_action_coordinates(
            completed_payloads
        )
        state["coordinate_audit"] = audit
        return original_membership(
            experiment,
            plan,
            rows_by_actor,
            normalized,
            orphan_record_ids,
        )

    def repaired_complete(experiment, plan, counts):
        result = original_complete(experiment, plan, counts)
        state["base_analysis_sha256"] = (
            frozen_formal.canonical_json_sha256(result)
        )
        if decorator is not None:
            result = decorator(result, state)
        return result

    frozen_analysis.validate_result_membership = repaired_membership
    frozen_analysis.analyze_complete = repaired_complete
    try:
        yield state
    finally:
        frozen_analysis.validate_result_membership = original_membership
        frozen_analysis.analyze_complete = original_complete


def coordinate_audit_actual(
    experiment: frozen_followup.AndroidRecoveredSourceFollowupExperiment,
) -> tuple[
    dict[str, dict[str, int]],
    dict[str, Any],
    dict[str, dict[str, int]],
]:
    with frozen_analysis.held_analysis_boundary(experiment):
        plan, counts = frozen_analysis.completion_preflight(experiment)
        (
            _,
            payloads,
            _,
            _,
            _,
        ) = frozen_analysis.validate_all_attempt_artifacts(experiment, plan)
        _, audit = normalize_integral_action_coordinates(payloads)
        snapshot = frozen_analysis.analysis_stability_snapshot(experiment)
    return audit, snapshot, counts


def legacy_failure(
    experiment: frozen_followup.AndroidRecoveredSourceFollowupExperiment,
) -> dict[str, Any]:
    try:
        frozen_analysis.analyze_with_boundary(experiment, output=None)
    except RuntimeError as error:
        if str(error) != "executed episode action schema changed":
            raise
        return {
            "error_type": type(error).__name__,
            "error": str(error),
            "stage": "validate_result_membership",
            "before_first_target_outcome_aggregation": True,
        }
    raise RuntimeError("legacy analyzer unexpectedly accepted float coordinates")


def validate_code_bindings(payload: dict[str, Any]) -> None:
    bindings = payload.get("bindings", {}).get("analysis_code", {})
    if not isinstance(bindings, dict) or not bindings:
        raise RuntimeError("repair code bindings are absent")
    for relative, expected in bindings.items():
        path = PROJECT / str(relative)
        if frozen_formal.sha256(path) != expected:
            raise RuntimeError(f"repair code binding changed: {relative}")


def validate_repair_materials() -> tuple[dict[str, Any], dict[str, Any]]:
    repair = strict_json_object(REPAIR_PATH)
    qualification = strict_json_object(QUALIFICATION_PATH)
    if (
        repair.get("repair_version") != REPAIR_VERSION
        or repair.get("paper_evidence") is not True
        or repair.get("post_result") is not True
        or repair.get("result_blind") is not False
        or repair.get("signed_analysis_existed_at_freeze") is not False
        or repair.get("frozen_android_head") != FROZEN_HEAD
        or repair.get("frozen_manifest_sha256") != FROZEN_MANIFEST_SHA256
    ):
        raise RuntimeError("integral-coordinate repair receipt is invalid")
    validate_code_bindings(repair)
    criteria = qualification.get("criteria")
    if (
        qualification.get("qualification_version") != QUALIFICATION_VERSION
        or qualification.get("paper_evidence") is not True
        or qualification.get("pass") is not True
        or not isinstance(criteria, dict)
        or not criteria
        or not all(criteria.values())
        or qualification.get("repair_sha256")
        != frozen_formal.sha256(REPAIR_PATH)
        or qualification.get("signed_analysis_existed") is not False
    ):
        raise RuntimeError("integral-coordinate repair qualification is invalid")
    validate_code_bindings(qualification)
    return repair, qualification


def final_decorator(
    repair: dict[str, Any],
    qualification: dict[str, Any],
) -> Decorator:
    def decorate(result: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
        if (
            state.get("coordinate_audit")
            != qualification.get("coordinate_audit")
            or state.get("base_analysis_sha256")
            != qualification.get("base_analysis_sha256")
        ):
            raise RuntimeError("qualified repaired analysis changed")
        result["analysis_version"] = ANALYSIS_VERSION
        result["integral_coordinate_schema_repair"] = {
            "version": REPAIR_VERSION,
            "path": REPAIR_PATH.relative_to(PROJECT).as_posix(),
            "sha256": frozen_formal.sha256(REPAIR_PATH),
            "qualification_version": QUALIFICATION_VERSION,
            "qualification_path": QUALIFICATION_PATH.relative_to(
                PROJECT
            ).as_posix(),
            "qualification_sha256": frozen_formal.sha256(QUALIFICATION_PATH),
            "legacy_failure": repair["legacy_failure"],
            "coordinate_audit": state["coordinate_audit"],
            "scope": repair["scope"],
            "required_disclosure": repair["required_disclosure"],
            "target_artifacts_modified": False,
            "normalization_target": "in-memory validation copy only",
        }
        result["analysis_boundary"]["coordinate_representation_repair"] = (
            "Post-result loader repair only: the bound runner serializes "
            "executed integer-valued pixels as JSON floats; the adapter accepts "
            "only finite integral numeric pairs, checks the executed-action "
            "reverse binding, and normalizes an in-memory validation copy. "
            "No episode artifact, outcome, policy, estimand, or claim gate changed."
        )
        return result

    return decorate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(FROZEN_MANIFEST))
    parser.add_argument("--output", default=str(ANALYSIS_OUTPUT))
    args = parser.parse_args()
    manifest = Path(args.manifest)
    output = Path(args.output)
    if str(manifest) != str(FROZEN_MANIFEST):
        raise RuntimeError("exact frozen Android V2 manifest is required")
    if str(output) != str(ANALYSIS_OUTPUT):
        raise RuntimeError("exact formal Android V2 analysis output is required")
    validate_frozen_checkout()
    if git(PROJECT, "status", "--porcelain=v1", "--untracked-files=all") != "":
        raise RuntimeError("repaired analysis requires a clean repair checkout")
    repair, qualification = validate_repair_materials()
    experiment = frozen_followup.AndroidRecoveredSourceFollowupExperiment(
        manifest
    )
    with use_integral_coordinate_repair(
        final_decorator(repair, qualification)
    ) as state:
        result = frozen_analysis.analyze_with_boundary(
            experiment,
            output=output,
        )
    print(
        json.dumps(
            {
                "output": str(output),
                "sha256": frozen_formal.sha256(output),
                "scope": result["scope"],
                "completion": result["completion"],
                "coordinate_audit": state["coordinate_audit"],
                "repair_version": REPAIR_VERSION,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
