"""Tail-aware dual-certificate v5 visual action-contract selector.

V5 preserves v4 except for a target-free nested-family underfit alarm on the
ordinary v2-supported path.  If the selected family's held-out P90 exceeds the
existing 6 px validation tolerance, a supported higher family reduces that
tail to at most 3 px while expanding source coverage by at least 0.10, and the
selected map also fails dense reconstruction, v5 refits the lowest such higher
family.  The upgrade is accepted only when its dense reconstruction passes.

The rule addresses model-family underfitting without imposing the deliberately
strict dense certificate on every otherwise valid global map.  Low-coverage
rescue remains v4's geometric-stability plus dense-photometric dual certificate.
No instruction, target box, or generated transform is consumed.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

import dafc.registration.base as v1
import dafc.registration.selector_v3 as v3
import dafc.registration.selector_v4 as v4
from dafc.registration.types import PointMatcher

FAMILY_ORDER = ("similarity", "affine", "homography")
SELECTED_VALIDATION_P90_UNDERFIT_PX = 6.0
HIGHER_VALIDATION_P90_GOOD_PX = 3.0
MINIMUM_HIGHER_SOURCE_COVERAGE_GAIN = 0.1
SELECTOR_RNG_SEED = v4.SELECTOR_RNG_SEED


def escalation_thresholds() -> dict[str, float]:
    return {
        "selected_validation_p90_underfit_px": SELECTED_VALIDATION_P90_UNDERFIT_PX,
        "higher_validation_p90_good_px": HIGHER_VALIDATION_P90_GOOD_PX,
        "minimum_higher_source_coverage_gain": MINIMUM_HIGHER_SOURCE_COVERAGE_GAIN,
    }


def higher_family_candidate(result: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    if not _eligible_for_escalation(result):
        return None
    selected_family = result["selected_family"]
    families = result.get("families") or {}
    selected = families.get(selected_family) or {}
    selected_p90 = selected.get("validation_p90_error")
    selected_coverage = selected.get("source_coverage")
    if (
        selected_p90 is None
        or selected_coverage is None
        or selected_p90 <= SELECTED_VALIDATION_P90_UNDERFIT_PX
    ):
        return None
    selected_index = FAMILY_ORDER.index(selected_family)
    for family in FAMILY_ORDER[selected_index + 1 :]:
        metrics = families.get(family) or {}
        validation_p90 = metrics.get("validation_p90_error")
        coverage = metrics.get("source_coverage")
        if (
            metrics.get("supported") is True
            and validation_p90 is not None
            and (validation_p90 <= HIGHER_VALIDATION_P90_GOOD_PX)
            and (coverage is not None)
            and (coverage - selected_coverage >= MINIMUM_HIGHER_SOURCE_COVERAGE_GAIN)
        ):
            return (family, metrics)
    return None


def with_v5_metadata(result: dict[str, Any], escalation: dict[str, Any]) -> dict[str, Any]:
    return {
        **result,
        "selector_version": "tail-aware-dual-certificate-v5",
        "parent_selector_version": result.get("selector_version"),
        "tail_family_escalation": escalation,
    }


def register_contract(
    source: np.ndarray, target: np.ndarray, *, matcher: PointMatcher | None = None
) -> dict[str, Any]:
    cv2.setNumThreads(1)
    cv2.setRNGSeed(SELECTOR_RNG_SEED)
    base = v4.register_contract(source, target, matcher=matcher)
    candidate = higher_family_candidate(base)
    if candidate is None:
        return with_v5_metadata(base, {"triggered": False, "thresholds": escalation_thresholds()})
    higher_family, higher_metrics = candidate
    selected_family = base["selected_family"]
    selected_metrics = base["families"][selected_family]
    source_points, target_points, counts = (matcher or v1.mutual_ratio_matches)(source, target)
    matrix, inliers = v1.fit_family(higher_family, source_points, target_points)
    escalation: dict[str, Any] = {
        "triggered": True,
        "thresholds": escalation_thresholds(),
        "from_family": selected_family,
        "to_family": higher_family,
        "selected_validation_p90_error": selected_metrics["validation_p90_error"],
        "higher_validation_p90_error": higher_metrics["validation_p90_error"],
        "selected_source_coverage": selected_metrics["source_coverage"],
        "higher_source_coverage": higher_metrics["source_coverage"],
        "match_counts": counts,
    }
    if matrix is None:
        return with_v5_metadata(
            {
                **base,
                "supported": False,
                "reason": "tail_family_escalation_refit_failed",
                "decision_path": "tail_aware_family_escalation",
                "candidate_matrix_audit": None,
                "matrix": None,
            },
            {**escalation, "fit": False, "dense_certificate_pass": False},
        )
    full_refit = {
        "matches": int(len(source_points)),
        "inliers": int(inliers.sum()),
        "inlier_rate": float(inliers.mean()),
        "source_coverage": v1.coverage_fraction(source_points[inliers], source.shape[1], source.shape[0]),
    }
    certificate = v4.photometric_certificate(source, target, matrix)
    gross_gate = v3.gross_photometric_gate(certificate)
    accepted = certificate["pass"] is True
    return with_v5_metadata(
        {
            **base,
            "supported": accepted,
            "selected_family": higher_family,
            "reason": "validation_tail_underfit_upgraded_and_dense_certified"
            if accepted
            else "validation_tail_underfit_upgrade_dense_certificate_failed",
            "decision_path": "tail_aware_family_escalation",
            "full_refit": full_refit,
            "photometric_certificate": certificate,
            "gross_photometric_gate": gross_gate,
            "candidate_matrix_audit": matrix.tolist(),
            "matrix": matrix.tolist() if accepted else None,
        },
        {
            **escalation,
            "fit": True,
            "full_refit": full_refit,
            "dense_certificate_pass": bool(certificate["pass"]),
            "accepted": accepted,
        },
    )


def _eligible_for_escalation(result):
    return not (
        result.get("supported") is not True
        or result.get("base_v2_supported") is not True
        or result.get("selected_family") not in FAMILY_ORDER
        or (result.get("decision_path") != "v2_supported_with_gross_photometric_gate")
        or ((result.get("gross_photometric_gate") or {}).get("pass") is not True)
        or ((result.get("photometric_certificate") or {}).get("pass") is True)
    )
