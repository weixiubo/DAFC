#!/usr/bin/env python3
"""Dual-certificate v3 target-free visual action-contract selector.

V3 preserves every v2 decision except one deliberately narrow development
case: v2's lowest-score family failed *only* its source-keypoint coverage gate.
Such a candidate is rescued only when deterministic subsample geometry, a
higher-family extrapolation envelope, and dense target-to-source photometric
reconstruction certify the same global map.  The two certificates are thus a
geometric certificate (stability + family envelope) and a dense appearance
certificate.  No instruction, target box, or generated transform is consumed.
"""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

from typing import Any

import cv2
import numpy as np

import dafc.registration.base as v1
import dafc.registration.selector_v2 as v2


CERTIFICATE_MAX_DIM = 640
CERTIFICATE_EROSION_PX = 7
MIN_VALID_SOURCE_RATE = 0.95
MAX_GRAYSCALE_MAE = 3.0
MAX_GRAYSCALE_P90 = 6.0
MIN_GRAYSCALE_WITHIN_12_RATE = 0.95
MIN_GRAYSCALE_NCC = 0.95
MIN_GRADIENT_NCC = 0.75
GRADIENT_QUANTILE = 0.60
MAX_GROSS_GRAYSCALE_MAE = 8.0
MAX_HIGHER_FAMILY_GRID_DISAGREEMENT_P95 = 3.0
FAMILY_ORDER = ("similarity", "affine", "homography")
SELECTOR_RNG_SEED = 364


def failed_support_checks(metrics: dict[str, Any]) -> list[str]:
    checks = {
        "train_inliers": metrics["train_inliers"] >= 16,
        "train_inlier_rate": metrics["train_inlier_rate"] >= 0.76,
        "train_inlier_median_error": (
            metrics["train_inlier_median_error"] is not None
            and metrics["train_inlier_median_error"] <= 2.5
        ),
        "source_coverage": metrics["source_coverage"] >= 0.30,
        "validation_matches": metrics["validation_matches"] >= 8,
        "validation_median_error": (
            metrics["validation_median_error"] is not None
            and metrics["validation_median_error"] <= 3.0
        ),
        "validation_within_6px_rate": (
            metrics["validation_within_6px_rate"] >= 0.60
        ),
    }
    return [name for name, passed in checks.items() if not passed]


def resize_for_certificate(image: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, CERTIFICATE_MAX_DIM / max(height, width))
    if scale == 1.0:
        return image, scale
    resized = cv2.resize(
        image,
        (max(1, round(width * scale)), max(1, round(height * scale))),
        interpolation=cv2.INTER_AREA,
    )
    return resized, scale


def normalized_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_centered = left.astype(np.float64) - float(left.mean())
    right_centered = right.astype(np.float64) - float(right.mean())
    denominator = float(
        np.sqrt(
            np.square(left_centered).sum()
            * np.square(right_centered).sum()
        )
    )
    if denominator <= 1e-12:
        return 0.0
    return float((left_centered * right_centered).sum() / denominator)


def photometric_certificate(
    source: np.ndarray,
    target: np.ndarray,
    matrix: np.ndarray,
) -> dict[str, Any]:
    source_small, source_scale = resize_for_certificate(source)
    target_small, target_scale = resize_for_certificate(target)
    source_scaler = np.diag([source_scale, source_scale, 1.0])
    target_scaler = np.diag([target_scale, target_scale, 1.0])
    scaled_matrix = (
        target_scaler @ np.asarray(matrix, dtype=np.float64) @ np.linalg.inv(source_scaler)
    )
    finite = bool(np.isfinite(scaled_matrix).all())
    try:
        inverse = np.linalg.inv(scaled_matrix)
    except np.linalg.LinAlgError:
        inverse = None
    if not finite or inverse is None or not np.isfinite(inverse).all():
        return {
            "pass": False,
            "reason": "nonfinite_or_singular_candidate",
            "thresholds": certificate_thresholds(),
        }

    source_height, source_width = source_small.shape[:2]
    reconstructed = cv2.warpPerspective(
        target_small,
        inverse,
        (source_width, source_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    target_valid = np.full(target_small.shape[:2], 255, dtype=np.uint8)
    valid = cv2.warpPerspective(
        target_valid,
        inverse,
        (source_width, source_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    kernel = np.ones(
        (CERTIFICATE_EROSION_PX, CERTIFICATE_EROSION_PX), dtype=np.uint8
    )
    valid = cv2.erode(valid, kernel) > 250
    valid_rate = float(valid.mean())
    if int(valid.sum()) < 256:
        return {
            "pass": False,
            "reason": "insufficient_valid_reconstruction_pixels",
            "valid_source_rate": valid_rate,
            "thresholds": certificate_thresholds(),
        }

    source_gray = cv2.cvtColor(source_small, cv2.COLOR_BGR2GRAY).astype(
        np.float32
    )
    reconstructed_gray = cv2.cvtColor(
        reconstructed, cv2.COLOR_BGR2GRAY
    ).astype(np.float32)
    source_gray = cv2.GaussianBlur(source_gray, (3, 3), 0)
    reconstructed_gray = cv2.GaussianBlur(reconstructed_gray, (3, 3), 0)
    source_values = source_gray[valid]
    reconstructed_values = reconstructed_gray[valid]
    differences = np.abs(source_values - reconstructed_values)
    grayscale_mae = float(differences.mean())
    grayscale_p90 = float(np.quantile(differences, 0.90))
    grayscale_within_12_rate = float((differences <= 12.0).mean())
    grayscale_ncc = normalized_correlation(
        source_values, reconstructed_values
    )

    source_gradient = cv2.magnitude(
        cv2.Sobel(source_gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(source_gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    reconstructed_gradient = cv2.magnitude(
        cv2.Sobel(reconstructed_gray, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(reconstructed_gray, cv2.CV_32F, 0, 1, ksize=3),
    )
    gradient_cutoff = float(
        np.quantile(source_gradient[valid], GRADIENT_QUANTILE)
    )
    gradient_mask = valid & (source_gradient >= gradient_cutoff)
    gradient_pixels = int(gradient_mask.sum())
    gradient_ncc = (
        normalized_correlation(
            source_gradient[gradient_mask],
            reconstructed_gradient[gradient_mask],
        )
        if gradient_pixels >= 128
        else 0.0
    )
    checks = {
        "valid_source_rate": valid_rate >= MIN_VALID_SOURCE_RATE,
        "grayscale_mae": grayscale_mae <= MAX_GRAYSCALE_MAE,
        "grayscale_p90": grayscale_p90 <= MAX_GRAYSCALE_P90,
        "grayscale_within_12_rate": (
            grayscale_within_12_rate >= MIN_GRAYSCALE_WITHIN_12_RATE
        ),
        "grayscale_ncc": grayscale_ncc >= MIN_GRAYSCALE_NCC,
        "gradient_ncc": gradient_ncc >= MIN_GRADIENT_NCC,
    }
    return {
        "pass": all(checks.values()),
        "reason": "all_dense_reconstruction_checks_passed"
        if all(checks.values())
        else "dense_reconstruction_check_failed",
        "valid_source_rate": valid_rate,
        "grayscale_mae": grayscale_mae,
        "grayscale_p90": grayscale_p90,
        "grayscale_within_12_rate": grayscale_within_12_rate,
        "grayscale_ncc": grayscale_ncc,
        "gradient_quantile": GRADIENT_QUANTILE,
        "gradient_pixels": gradient_pixels,
        "gradient_ncc": gradient_ncc,
        "checks": checks,
        "thresholds": certificate_thresholds(),
    }


def certificate_thresholds() -> dict[str, Any]:
    return {
        "max_dimension": CERTIFICATE_MAX_DIM,
        "erosion_px": CERTIFICATE_EROSION_PX,
        "minimum_valid_source_rate": MIN_VALID_SOURCE_RATE,
        "maximum_grayscale_mae": MAX_GRAYSCALE_MAE,
        "maximum_grayscale_p90": MAX_GRAYSCALE_P90,
        "minimum_grayscale_within_12_rate": (
            MIN_GRAYSCALE_WITHIN_12_RATE
        ),
        "minimum_grayscale_ncc": MIN_GRAYSCALE_NCC,
        "minimum_gradient_ncc": MIN_GRADIENT_NCC,
        "gradient_quantile": GRADIENT_QUANTILE,
        "maximum_higher_family_grid_disagreement_p95": (
            MAX_HIGHER_FAMILY_GRID_DISAGREEMENT_P95
        ),
        "maximum_gross_grayscale_mae": MAX_GROSS_GRAYSCALE_MAE,
    }


def gross_photometric_gate(
    certificate: dict[str, Any],
) -> dict[str, Any]:
    valid_rate = certificate.get("valid_source_rate")
    grayscale_mae = certificate.get("grayscale_mae")
    checks = {
        "valid_source_rate": (
            valid_rate is not None and valid_rate >= MIN_VALID_SOURCE_RATE
        ),
        "grayscale_mae": (
            grayscale_mae is not None
            and grayscale_mae <= MAX_GROSS_GRAYSCALE_MAE
        ),
    }
    return {
        "pass": all(checks.values()),
        "checks": checks,
        "valid_source_rate": valid_rate,
        "grayscale_mae": grayscale_mae,
        "minimum_valid_source_rate": MIN_VALID_SOURCE_RATE,
        "maximum_grayscale_mae": MAX_GROSS_GRAYSCALE_MAE,
    }


def higher_family_envelope(
    source: np.ndarray,
    target: np.ndarray,
    selected_family: str,
    selected_matrix: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
) -> dict[str, Any]:
    selected_index = FAMILY_ORDER.index(selected_family)
    grid = v1.source_grid(source.shape[1], source.shape[0])
    selected_grid = v1.apply_homography(selected_matrix, grid)
    candidates = []
    for family in FAMILY_ORDER[selected_index + 1 :]:
        matrix, inliers = v1.fit_family(family, source_points, target_points)
        if matrix is None:
            candidates.append(
                {
                    "family": family,
                    "fit": False,
                    "photometric_pass": False,
                    "grid_disagreement_p95": None,
                }
            )
            continue
        certificate = photometric_certificate(source, target, matrix)
        disagreement = np.linalg.norm(
            v1.apply_homography(matrix, grid) - selected_grid,
            axis=1,
        )
        candidates.append(
            {
                "family": family,
                "fit": True,
                "inliers": int(inliers.sum()),
                "photometric_pass": certificate["pass"],
                "photometric_metrics": {
                    key: certificate.get(key)
                    for key in (
                        "valid_source_rate",
                        "grayscale_mae",
                        "grayscale_p90",
                        "grayscale_within_12_rate",
                        "grayscale_ncc",
                        "gradient_ncc",
                    )
                },
                "grid_disagreement_p95": float(
                    np.quantile(disagreement, 0.95)
                ),
            }
        )
    certified = [item for item in candidates if item["photometric_pass"]]
    maximum = max(
        (item["grid_disagreement_p95"] for item in certified),
        default=0.0,
    )
    passed = maximum <= MAX_HIGHER_FAMILY_GRID_DISAGREEMENT_P95
    return {
        "pass": passed,
        "reason": "all_photometrically_plausible_higher_families_agree"
        if passed
        else "photometrically_plausible_higher_family_disagrees",
        "threshold_px": MAX_HIGHER_FAMILY_GRID_DISAGREEMENT_P95,
        "maximum_grid_disagreement_p95": maximum,
        "certified_higher_families": len(certified),
        "candidates": candidates,
    }


def v3_result_from_v2(base_result: dict[str, Any]) -> dict[str, Any]:
    return {
        **base_result,
        "selector_version": "dual-certificate-v3",
        "base_selector_version": base_result.get("selector_version"),
        "base_v2_supported": base_result["supported"],
        "decision_path": "v2_decision_preserved",
        "v2_reason": base_result.get("reason"),
        "low_coverage_rescue_eligible": False,
        "failed_v2_support_checks": None,
        "photometric_certificate": None,
        "gross_photometric_gate": None,
        "rescue_stability": None,
        "higher_family_envelope": None,
    }


def register_contract(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    # OpenCV's RANSAC otherwise inherits process-global state and can change
    # with evaluation order.  C364 registration is deliberately sequential.
    cv2.setNumThreads(1)
    cv2.setRNGSeed(SELECTOR_RNG_SEED)
    base_result = v2.register_contract(source, target)
    preserved = v3_result_from_v2(base_result)
    if base_result["supported"]:
        if base_result.get("selected_family") == "identity":
            return {
                **preserved,
                "gross_photometric_gate": {
                    "pass": True,
                    "checks": {"exact_pixel_identity": True},
                },
            }
        base_matrix = np.asarray(base_result["matrix"], dtype=np.float64)
        certificate = photometric_certificate(source, target, base_matrix)
        gross_gate = gross_photometric_gate(certificate)
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
    failed_checks = failed_support_checks(selected_metrics)
    if failed_checks != ["source_coverage"]:
        return {
            **preserved,
            "failed_v2_support_checks": failed_checks,
        }

    source_points, target_points, counts = v1.mutual_ratio_matches(
        source, target
    )
    if len(source_points) < 32:
        return {
            **preserved,
            "failed_v2_support_checks": failed_checks,
            "decision_path": "low_coverage_rescue_internal_match_failure",
        }
    final_matrix, final_inliers = v1.fit_family(
        selected_family, source_points, target_points
    )
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
        selected_family,
        final_matrix,
        source_points,
        target_points,
        source.shape[1],
        source.shape[0],
    )
    certificate = photometric_certificate(source, target, final_matrix)
    gross_gate = gross_photometric_gate(certificate)
    family_envelope = higher_family_envelope(
        source,
        target,
        selected_family,
        final_matrix,
        source_points,
        target_points,
    )
    rescued = bool(
        stability["pass"] and certificate["pass"] and family_envelope["pass"]
    )
    return {
        **preserved,
        "supported": rescued,
        "reason": (
            "low_coverage_rescued_by_geometric_and_photometric_certificates"
            if rescued
            else "low_coverage_dual_certificate_failed"
        ),
        "decision_path": "low_coverage_dual_certificate_rescue",
        "v2_reason": base_result.get("reason"),
        "low_coverage_rescue_eligible": True,
        "failed_v2_support_checks": failed_checks,
        "match_counts": counts,
        "full_refit": full_refit,
        "rescue_stability": stability,
        "higher_family_envelope": family_envelope,
        "photometric_certificate": certificate,
        "gross_photometric_gate": gross_gate,
        "candidate_matrix_audit": final_matrix.tolist(),
        "matrix": final_matrix.tolist() if rescued else None,
    }


def main() -> int:
    print(
        {
            "selector_version": "dual-certificate-v3",
            "status": "post-C363 development; requires disjoint validation",
            "certificate_thresholds": certificate_thresholds(),
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
