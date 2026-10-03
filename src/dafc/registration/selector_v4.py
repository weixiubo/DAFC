"""Simplified dual-certificate v4 visual action-contract selector.

V4 keeps v3's target-free candidate fitting and all-path gross photometric
gate.  Its only change is the low-coverage rescue rule selected by the frozen
C371 mechanism analysis: a candidate must pass deterministic subsample
geometric stability and dense photometric reconstruction.  The higher-family
envelope is removed because it was strictly dominated by this two-certificate
rule on C371.  C366/C371 are therefore development evidence for v4, not an
independent confirmation of it.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

import dafc.registration.base as v1
import dafc.registration.selector_v2 as v2
import dafc.registration.selector_v3 as v3
from dafc.registration.types import PointMatcher

SELECTOR_RNG_SEED = v3.SELECTOR_RNG_SEED


def certificate_thresholds() -> dict[str, Any]:
    thresholds = dict(v3.certificate_thresholds())
    thresholds.pop("maximum_higher_family_grid_disagreement_p95")
    return thresholds


def photometric_certificate(source: np.ndarray, target: np.ndarray, matrix: np.ndarray) -> dict[str, Any]:
    certificate = v3.photometric_certificate(source, target, matrix)
    return {**certificate, "thresholds": certificate_thresholds()}


def v4_result_from_v2(base_result: dict[str, Any]) -> dict[str, Any]:
    return {
        **base_result,
        "selector_version": "dual-certificate-v4",
        "base_selector_version": base_result.get("selector_version"),
        "base_v2_supported": base_result["supported"],
        "decision_path": "v2_decision_preserved",
        "v2_reason": base_result.get("reason"),
        "low_coverage_rescue_eligible": False,
        "failed_v2_support_checks": None,
        "photometric_certificate": None,
        "gross_photometric_gate": None,
        "rescue_stability": None,
    }


def register_contract(
    source: np.ndarray, target: np.ndarray, *, matcher: PointMatcher | None = None
) -> dict[str, Any]:
    cv2.setNumThreads(1)
    cv2.setRNGSeed(SELECTOR_RNG_SEED)
    base_result = v2.register_contract(source, target, matcher=matcher)
    preserved = v4_result_from_v2(base_result)
    if base_result["supported"]:
        if base_result.get("selected_family") == "identity":
            return {
                **preserved,
                "gross_photometric_gate": {"pass": True, "checks": {"exact_pixel_identity": True}},
            }
        base_matrix = np.asarray(base_result["matrix"], dtype=np.float64)
        certificate = photometric_certificate(source, target, base_matrix)
        gross_gate = v3.gross_photometric_gate(certificate)
        if gross_gate["pass"]:
            return {
                **preserved,
                "decision_path": "v2_supported_with_gross_photometric_gate",
                "photometric_certificate": certificate,
                "gross_photometric_gate": gross_gate,
            }
        return {
            **preserved,
            "supported": False,
            "reason": "v2_supported_rejected_by_gross_photometric_gate",
            "decision_path": "v2_supported_with_gross_photometric_gate",
            "photometric_certificate": certificate,
            "gross_photometric_gate": gross_gate,
            "candidate_matrix_audit": base_result["matrix"],
            "matrix": None,
        }
    if base_result.get("reason") != "lowest_score_family_lacks_global_support":
        return preserved
    selected_metrics = base_result.get("selected_family_support")
    selected_family = base_result.get("selected_family")
    if not isinstance(selected_metrics, dict) or selected_family not in (
        "similarity",
        "affine",
        "homography",
    ):
        return preserved
    failed_checks = v3.failed_support_checks(selected_metrics)
    if failed_checks != ["source_coverage"]:
        return {**preserved, "failed_v2_support_checks": failed_checks}
    source_points, target_points, counts = (matcher or v1.mutual_ratio_matches)(source, target)
    if len(source_points) < 32:
        return {
            **preserved,
            "failed_v2_support_checks": failed_checks,
            "decision_path": "low_coverage_rescue_internal_match_failure",
        }
    final_matrix, final_inliers = v1.fit_family(selected_family, source_points, target_points)
    if final_matrix is None:
        return {
            **preserved,
            "failed_v2_support_checks": failed_checks,
            "decision_path": "low_coverage_rescue_full_refit_failure",
        }
    full_refit = {
        "matches": int(len(source_points)),
        "inliers": int(final_inliers.sum()),
        "inlier_rate": float(final_inliers.mean()),
        "source_coverage": v1.coverage_fraction(
            source_points[final_inliers], source.shape[1], source.shape[0]
        ),
    }
    stability = v2.fit_stability(
        selected_family, final_matrix, source_points, target_points, source.shape[1], source.shape[0]
    )
    certificate = photometric_certificate(source, target, final_matrix)
    gross_gate = v3.gross_photometric_gate(certificate)
    rescued = bool(stability["pass"] and certificate["pass"])
    return {
        **preserved,
        "supported": rescued,
        "reason": "low_coverage_rescued_by_geometric_and_photometric_certificates"
        if rescued
        else "low_coverage_dual_certificate_failed",
        "decision_path": "low_coverage_dual_certificate_rescue",
        "v2_reason": base_result.get("reason"),
        "low_coverage_rescue_eligible": True,
        "failed_v2_support_checks": failed_checks,
        "match_counts": counts,
        "full_refit": full_refit,
        "rescue_stability": stability,
        "photometric_certificate": certificate,
        "gross_photometric_gate": gross_gate,
        "candidate_matrix_audit": final_matrix.tolist(),
        "matrix": final_matrix.tolist() if rescued else None,
    }
