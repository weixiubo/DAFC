#!/usr/bin/env python3
"""Verify and materialize the frozen C398 browser-native V6 protocol."""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


from dafc.paths import PROJECT
PROTOCOL = PROJECT / "reports" / "c398_browser_native_v6_preregistration.md"
SELECTOR = PROJECT / "scripts" / "c390_action_conditioned_selector_v6.py"
V5 = PROJECT / "scripts" / "c375_visual_contract_selector_v5.py"
RUNNER = PROJECT / "scripts" / "remote_open_gui_grounder.py"
CONTRACT_SOURCE = PROJECT / "scripts" / "c388_browser_native_contract_development_pilot.py"
METHOD_CONFIRMATION = PROJECT / "validation" / "c394_corrected_click_harness_replay.json"
MODEL_CONFIRMATION = PROJECT / "validation" / "c397_remote_open_models_decoder_confirmation.json"
OUT_JSON = PROJECT / "validation" / "c398_browser_native_v6_preregistration.json"

EXPECTED_HASHES = {
    "scripts/c390_action_conditioned_selector_v6.py": "c7dff3cb101e2b232d1bea39864220c1e5c12138aa348aa1a32b6fb43ee075a5",
    "scripts/c375_visual_contract_selector_v5.py": "148135d50f5ecc18cc3127deebc232750ea0f3c2054368f1aa09c4514d2fa8d4",
    "scripts/remote_open_gui_grounder.py": "dcb71f6c6be498502eacc451a56ccc135828e8e4cbd6a528d611fe76ff93ace4",
    "scripts/c388_browser_native_contract_development_pilot.py": "423765cde082d9b6fd4b95520a3d02eb920a7183004861792a9f0d029b16aa0e",
    "validation/c394_corrected_click_harness_replay.json": "68c0b252e512f569abadc6c84d429eb61f26fe5e9c450755cc389510d9840e29",
    "validation/c397_remote_open_models_decoder_confirmation.json": "7a8acbcb36438acba42e68ec67f3f2c4d9b9783c71263130d529bbc90ee694c8",
}

TASKS = (
    "click-button",
    "click-color",
    "click-dialog",
    "click-dialog-2",
    "click-link",
    "click-tab",
    "click-test",
    "click-test-2",
    "click-test-transfer",
    "grid-coordinate",
)
CONTRACTS = (
    "native_zoom_125",
    "native_zoom_80",
    "native_affine_skew",
    "native_jpeg_stream",
    "native_narrow_reflow",
)
MODELS = (
    "uitars-2b-sft",
    "showui-2b",
    "qwen2.5-vl-7b-instruct",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(PROJECT).as_posix()


def git_blob_matches_head(path: Path) -> bool:
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel(path)],
        cwd=PROJECT,
        capture_output=True,
        check=False,
    )
    if tracked.returncode != 0:
        return False
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", rel(path)],
        cwd=PROJECT,
        check=False,
    )
    return clean.returncode == 0


def final_seed(task: str, ordinal: int) -> int:
    payload = f"c398-final-seed-v1|{task}|{ordinal}".encode("utf-8")
    return 398_000_000 + int(hashlib.sha256(payload).hexdigest(), 16) % 500_000_000


def main() -> int:
    bound_paths = (
        PROTOCOL,
        Path(__file__).resolve(),
        SELECTOR,
        V5,
        RUNNER,
        CONTRACT_SOURCE,
        METHOD_CONFIRMATION,
        MODEL_CONFIRMATION,
    )
    committed = {rel(path): git_blob_matches_head(path) for path in bound_paths}
    observed_hashes = {
        rel(path): sha256(path)
        for path in (
            SELECTOR,
            V5,
            RUNNER,
            CONTRACT_SOURCE,
            METHOD_CONFIRMATION,
            MODEL_CONFIRMATION,
        )
    }
    method = json.loads(METHOD_CONFIRMATION.read_text(encoding="utf-8"))
    models = json.loads(MODEL_CONFIRMATION.read_text(encoding="utf-8"))
    checkpoint_bindings: dict[str, Any] = {
        row["model"]: {
            "kind": row["kind"],
            "config_sha256": row["checkpoint"]["config_sha256"],
            "total_weight_bytes": row["checkpoint"]["total_weight_bytes"],
            "weight_files": row["checkpoint"]["weight_files"],
        }
        for row in models["models"]
    }
    criteria = {
        "all_protocol_inputs_committed": all(committed.values()),
        "all_expected_hashes_match": observed_hashes == EXPECTED_HASHES,
        "method_confirmation_passed": method.get("pass") is True,
        "model_confirmation_passed": models.get("pass") is True,
        "exact_ten_tasks": len(TASKS) == 10,
        "exact_five_contracts": len(CONTRACTS) == 5,
        "exact_three_models": tuple(checkpoint_bindings) == MODELS,
        "development_seeds_excluded_from_preview": all(
            final_seed(task, ordinal) not in {387001, 387002, 387003}
            for task in TASKS
            for ordinal in range(16)
        ),
        "zero_final_qualification_report_before_preregistration": not (
            PROJECT / "validation" / "c399_final_browser_source_qualification.json"
        ).exists(),
    }
    result = {
        "title": "C398 Frozen Browser-Native V6 External-Validity Preregistration",
        "paper_evidence": False,
        "protocol": {
            "path": rel(PROTOCOL),
            "sha256": sha256(PROTOCOL),
        },
        "committed": committed,
        "expected_hashes": EXPECTED_HASHES,
        "observed_hashes": observed_hashes,
        "tasks": list(TASKS),
        "contracts": list(CONTRACTS),
        "models": list(MODELS),
        "checkpoint_bindings": checkpoint_bindings,
        "population": {
            "sources_per_task": 8,
            "sources": 80,
            "source_model_calls": 240,
            "target_trials": 1200,
            "target_model_calls": 1200,
        },
        "seed_selection": {
            "version": "c398-final-seed-v1",
            "candidate_limit_per_task": 256,
            "first_candidate_seeds": {
                task: [final_seed(task, ordinal) for ordinal in range(10)]
                for task in TASKS
            },
            "forbidden_development_seeds": [387001, 387002, 387003],
        },
        "primary": {
            "effect": "V6+fallback minus always-reground success",
            "noninferiority_margin": -0.03,
            "minimum_target_call_reduction": 0.30,
            "minimum_source_correct_accepted_transfer_success": 0.99,
            "minimum_model_stratum_difference": -0.05,
            "bootstrap_replicates": 20000,
            "rng_seed": 398001,
        },
        "criteria": criteria,
        "pass": all(criteria.values()),
    }
    OUT_JSON.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"criteria": criteria, "pass": result["pass"]}, indent=2))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
