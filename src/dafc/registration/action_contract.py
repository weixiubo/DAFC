"""Development candidate: action-conditioned DAFC v6 selector.

V6 preserves every V5 acceptance.  For V5 abstentions, it may reuse the
candidate map only when evidence local to the already grounded source action
is stable.  The rescue combines (i) deterministic mapped-action stability,
(ii) nearby inlier support or local appearance reconstruction, and (iii) a
non-degenerate local map.  It consumes source/target pixels and the source
action only; instruction text, target DOM, target action, and reward are never
inputs.

This file is a development candidate selected after C389.  It requires a new
disjoint-seed confirmation before it can be frozen for paper-facing use.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

from dafc.registration import base as V1
from dafc.registration import selector_v5 as V5
from dafc.registration.types import PointMatcher

FAMILIES = ("similarity", "affine", "homography")
RNG_SEED = 390
SUBSAMPLE_REPETITIONS = 7
SUBSAMPLE_KEEP_FRACTION = 0.8
PATCH_RADIUS_PX = 72
MINIMUM_MATCHES = 32
MINIMUM_INLIERS = 16
MINIMUM_INLIER_RATE = 0.7
MAXIMUM_INLIER_MEDIAN_RESIDUAL_PX = 2.5
MAXIMUM_INLIER_P90_RESIDUAL_PX = 4.0
MAXIMUM_ACTION_STABILITY_PX = 4.0
MINIMUM_LOCAL_SCALE = 0.35
MAXIMUM_LOCAL_SCALE = 2.5
MAXIMUM_LOCAL_CONDITION_NUMBER = 4.0
TARGET_BOUNDARY_MARGIN_PX = 4.0
GEOMETRIC_RADIUS_FRACTION = 0.12
GEOMETRIC_MINIMUM_INLIERS = 6
GEOMETRIC_MINIMUM_ANGULAR_BINS = 3
GEOMETRIC_MAXIMUM_LOCAL_P90_RESIDUAL_PX = 4.0
GEOMETRIC_MAXIMUM_NEAREST_INLIER_FRACTION = 0.12
GEOMETRIC_MAXIMUM_GRAYSCALE_MAE = 35.0
GEOMETRIC_MINIMUM_GRAYSCALE_NCC = 0.5
APPEARANCE_RADIUS_FRACTION = 0.18
APPEARANCE_MINIMUM_INLIERS = 1
APPEARANCE_MAXIMUM_NEAREST_INLIER_FRACTION = 0.16
APPEARANCE_MAXIMUM_GRAYSCALE_MAE = 5.0
APPEARANCE_MAXIMUM_GRAYSCALE_P90 = 10.0
APPEARANCE_MINIMUM_WITHIN_12_RATE = 0.95
APPEARANCE_MINIMUM_GRAYSCALE_NCC = 0.9
APPEARANCE_MINIMUM_GRADIENT_NCC = 0.75
MINIMUM_LOCAL_VALID_RATE = 0.9


def thresholds() -> dict[str, Any]:
    return {
        "minimum_matches": MINIMUM_MATCHES,
        "minimum_inliers": MINIMUM_INLIERS,
        "minimum_inlier_rate": MINIMUM_INLIER_RATE,
        "maximum_inlier_median_residual_px": MAXIMUM_INLIER_MEDIAN_RESIDUAL_PX,
        "maximum_inlier_p90_residual_px": MAXIMUM_INLIER_P90_RESIDUAL_PX,
        "subsample_repetitions": SUBSAMPLE_REPETITIONS,
        "subsample_keep_fraction": SUBSAMPLE_KEEP_FRACTION,
        "maximum_action_stability_px": MAXIMUM_ACTION_STABILITY_PX,
        "minimum_local_scale": MINIMUM_LOCAL_SCALE,
        "maximum_local_scale": MAXIMUM_LOCAL_SCALE,
        "maximum_local_condition_number": MAXIMUM_LOCAL_CONDITION_NUMBER,
        "target_boundary_margin_px": TARGET_BOUNDARY_MARGIN_PX,
        "patch_radius_px": PATCH_RADIUS_PX,
        "minimum_local_valid_rate": MINIMUM_LOCAL_VALID_RATE,
        "geometric_branch": {
            "radius_fraction": GEOMETRIC_RADIUS_FRACTION,
            "minimum_inliers": GEOMETRIC_MINIMUM_INLIERS,
            "minimum_angular_bins": GEOMETRIC_MINIMUM_ANGULAR_BINS,
            "maximum_local_p90_residual_px": GEOMETRIC_MAXIMUM_LOCAL_P90_RESIDUAL_PX,
            "maximum_nearest_inlier_fraction": GEOMETRIC_MAXIMUM_NEAREST_INLIER_FRACTION,
            "maximum_grayscale_mae": GEOMETRIC_MAXIMUM_GRAYSCALE_MAE,
            "minimum_grayscale_ncc": GEOMETRIC_MINIMUM_GRAYSCALE_NCC,
        },
        "appearance_branch": {
            "radius_fraction": APPEARANCE_RADIUS_FRACTION,
            "minimum_inliers": APPEARANCE_MINIMUM_INLIERS,
            "maximum_nearest_inlier_fraction": APPEARANCE_MAXIMUM_NEAREST_INLIER_FRACTION,
            "maximum_grayscale_mae": APPEARANCE_MAXIMUM_GRAYSCALE_MAE,
            "maximum_grayscale_p90": APPEARANCE_MAXIMUM_GRAYSCALE_P90,
            "minimum_grayscale_within_12_rate": APPEARANCE_MINIMUM_WITHIN_12_RATE,
            "minimum_grayscale_ncc": APPEARANCE_MINIMUM_GRAYSCALE_NCC,
            "minimum_gradient_ncc": APPEARANCE_MINIMUM_GRADIENT_NCC,
        },
    }


def as_action(source_action: np.ndarray | list[float] | tuple[float, float]) -> np.ndarray:
    action = np.asarray(source_action, dtype=np.float64).reshape(-1, 2)
    if len(action) != 1 or not np.isfinite(action).all():
        raise ValueError("source_action must contain one finite (x, y) point")
    return action


def deterministic_keep_mask(
    source_points: np.ndarray, target_points: np.ndarray, repetition: int
) -> np.ndarray:
    cutoff = round(SUBSAMPLE_KEEP_FRACTION * 10000)
    kept = []
    for index, (source, target) in enumerate(zip(source_points, target_points, strict=False)):
        key = (
            f"{RNG_SEED}:{repetition}:{index}:{source[0]:.3f}:{source[1]:.3f}:{target[0]:.3f}:{target[1]:.3f}"
        )
        kept.append(int(hashlib.sha256(key.encode()).hexdigest(), 16) % 10000 < cutoff)
    return np.asarray(kept, dtype=bool)


def action_stability(
    family: str,
    full_matrix: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    action: np.ndarray,
) -> dict[str, Any]:
    reference = V1.apply_homography(full_matrix, action)[0]
    details = []
    for repetition in range(SUBSAMPLE_REPETITIONS):
        mask = deterministic_keep_mask(source_points, target_points, repetition)
        subset_source = source_points[mask]
        subset_target = target_points[mask]
        if len(subset_source) < MINIMUM_MATCHES:
            details.append(
                {
                    "repetition": repetition,
                    "fit": False,
                    "reason": "fewer_than_minimum_subsample_matches",
                    "matches": int(len(subset_source)),
                }
            )
            continue
        cv2.setRNGSeed(RNG_SEED + repetition)
        try:
            matrix, inliers = V1.fit_family(family, subset_source, subset_target)
        except cv2.error:
            matrix, inliers = (None, np.zeros(len(subset_source), dtype=bool))
        if matrix is None or int(inliers.sum()) < MINIMUM_INLIERS:
            details.append(
                {
                    "repetition": repetition,
                    "fit": False,
                    "reason": "subsample_fit_or_inlier_failure",
                    "matches": int(len(subset_source)),
                    "inliers": int(inliers.sum()),
                }
            )
            continue
        mapped = V1.apply_homography(matrix, action)[0]
        disagreement = float(np.linalg.norm(mapped - reference))
        details.append(
            {
                "repetition": repetition,
                "fit": True,
                "matches": int(len(subset_source)),
                "inliers": int(inliers.sum()),
                "mapped_action": mapped.tolist(),
                "disagreement_px": disagreement,
            }
        )
    fitted = [item for item in details if item.get("fit")]
    maximum = max((item["disagreement_px"] for item in fitted)) if fitted else None
    passed = bool(
        len(fitted) == SUBSAMPLE_REPETITIONS
        and maximum is not None
        and (maximum <= MAXIMUM_ACTION_STABILITY_PX)
    )
    return {
        "pass": passed,
        "maximum_disagreement_px": maximum,
        "fitted_repetitions": len(fitted),
        "repetitions": SUBSAMPLE_REPETITIONS,
        "details": details,
    }


def normalized_correlation(left: np.ndarray, right: np.ndarray) -> float | None:
    left_centered = left.astype(np.float64) - float(left.mean())
    right_centered = right.astype(np.float64) - float(right.mean())
    denominator = float(np.sqrt(np.square(left_centered).sum() * np.square(right_centered).sum()))
    if denominator <= 1e-12:
        return None
    return float((left_centered * right_centered).sum() / denominator)


def local_photometric(
    source: np.ndarray, target: np.ndarray, matrix: np.ndarray, action: np.ndarray
) -> dict[str, Any]:
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return {"valid": False, "reason": "singular_matrix"}
    if not np.isfinite(inverse).all():
        return {"valid": False, "reason": "nonfinite_inverse"}
    height, width = source.shape[:2]
    reconstructed = cv2.warpPerspective(
        target,
        inverse,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(0, 0, 0),
    )
    target_valid = np.full(target.shape[:2], 255, dtype=np.uint8)
    valid = cv2.warpPerspective(
        target_valid,
        inverse,
        (width, height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    valid = cv2.erode(valid, np.ones((5, 5), dtype=np.uint8)) > 250
    center_x, center_y = action.reshape(2)
    yy, xx = np.ogrid[:height, :width]
    local = np.square(xx - center_x) + np.square(yy - center_y) <= float(PATCH_RADIUS_PX**2)
    mask = valid & local
    local_pixels = int(local.sum())
    valid_pixels = int(mask.sum())
    if valid_pixels < 256:
        return {
            "valid": False,
            "reason": "insufficient_valid_local_pixels",
            "valid_rate": float(valid_pixels / max(1, local_pixels)),
        }
    source_gray = cv2.GaussianBlur(cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0)
    target_gray = cv2.GaussianBlur(
        cv2.cvtColor(reconstructed, cv2.COLOR_BGR2GRAY).astype(np.float32), (3, 3), 0
    )
    source_values = source_gray[mask]
    target_values = target_gray[mask]
    differences = np.abs(source_values - target_values)
    source_gradient = cv2.magnitude(
        cv2.Sobel(source_gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(source_gray, cv2.CV_32F, 0, 1, ksize=3)
    )
    target_gradient = cv2.magnitude(
        cv2.Sobel(target_gray, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(target_gray, cv2.CV_32F, 0, 1, ksize=3)
    )
    gradient_cutoff = float(np.quantile(source_gradient[mask], 0.6))
    gradient_mask = mask & (source_gradient >= gradient_cutoff)
    gradient_pixels = int(gradient_mask.sum())
    return {
        "valid": True,
        "valid_rate": float(valid_pixels / local_pixels),
        "grayscale_mae": float(differences.mean()),
        "grayscale_p90": float(np.quantile(differences, 0.9)),
        "grayscale_within_12_rate": float((differences <= 12.0).mean()),
        "grayscale_ncc": normalized_correlation(source_values, target_values),
        "gradient_pixels": gradient_pixels,
        "gradient_ncc": normalized_correlation(source_gradient[gradient_mask], target_gradient[gradient_mask])
        if gradient_pixels >= 128
        else None,
    }


def angular_bins(points: np.ndarray, action: np.ndarray) -> int:
    if not len(points):
        return 0
    delta = points - action.reshape(2)
    return len(
        set(
            (np.floor((np.arctan2(delta[:, 1], delta[:, 0]) + math.pi) / (math.pi / 2)) % 4)
            .astype(int)
            .tolist()
        )
    )


def local_support(
    inlier_source: np.ndarray,
    inlier_residuals: np.ndarray,
    action: np.ndarray,
    source_diagonal: float,
    radius_fraction: float,
) -> dict[str, Any]:
    distances = np.linalg.norm(inlier_source - action.reshape(2), axis=1)
    mask = distances <= radius_fraction * source_diagonal
    return {
        "radius_fraction": radius_fraction,
        "radius_px": radius_fraction * source_diagonal,
        "inliers": int(mask.sum()),
        "angular_bins": angular_bins(inlier_source[mask], action),
        "residual_median_px": float(np.median(inlier_residuals[mask])) if mask.any() else None,
        "residual_p90_px": float(np.quantile(inlier_residuals[mask], 0.9)) if mask.any() else None,
    }


def local_map_shape(matrix: np.ndarray, action: np.ndarray) -> dict[str, Any]:
    probes = np.vstack([action, action + np.asarray([[1.0, 0.0]]), action + np.asarray([[0.0, 1.0]])])
    mapped = V1.apply_homography(matrix, probes)
    jacobian = np.column_stack((mapped[1] - mapped[0], mapped[2] - mapped[0]))
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    determinant = float(np.linalg.det(jacobian))
    minimum = float(np.min(singular_values))
    maximum = float(np.max(singular_values))
    condition = maximum / minimum if minimum > 1e-12 else math.inf
    passed = bool(
        determinant > 0.0
        and minimum >= MINIMUM_LOCAL_SCALE
        and (maximum <= MAXIMUM_LOCAL_SCALE)
        and (condition <= MAXIMUM_LOCAL_CONDITION_NUMBER)
    )
    return {
        "pass": passed,
        "determinant": determinant,
        "minimum_singular_value": minimum,
        "maximum_singular_value": maximum,
        "condition_number": condition,
    }


def mapped_action_in_bounds(mapped_action: np.ndarray, target_width: int, target_height: int) -> bool:
    x, y = mapped_action.reshape(2)
    return bool(
        TARGET_BOUNDARY_MARGIN_PX <= x < target_width - TARGET_BOUNDARY_MARGIN_PX
        and TARGET_BOUNDARY_MARGIN_PX <= y < target_height - TARGET_BOUNDARY_MARGIN_PX
    )


def register_action_contract(
    source: np.ndarray,
    target: np.ndarray,
    source_action: np.ndarray | list[float] | tuple[float, float],
    *,
    matcher: PointMatcher | None = None,
) -> dict[str, Any]:
    cv2.setNumThreads(1)
    cv2.setRNGSeed(RNG_SEED)
    action = as_action(source_action)
    v5 = V5.register_contract(source, target, matcher=matcher)
    if v5.get("supported"):
        return _preserve_parent(v5, action)
    selected_family = v5.get("selected_family")
    rescue: dict[str, Any] = {
        "eligible": selected_family in FAMILIES,
        "accepted": False,
        "reason": "no_eligible_v5_candidate_family",
        "thresholds": thresholds(),
        "source_action": action[0].tolist(),
        "v5_reason": v5.get("reason"),
        "v5_decision_path": v5.get("decision_path"),
        "selected_family": selected_family,
    }
    if selected_family not in FAMILIES:
        return {
            **v5,
            "selector_version": "action-conditioned-v6-development",
            "parent_selector_version": v5.get("selector_version"),
            "action_conditioned_rescue": rescue,
        }
    source_points, target_points, counts = (matcher or V1.mutual_ratio_matches)(source, target)
    rescue["match_counts"] = counts
    if len(source_points) < MINIMUM_MATCHES:
        rescue["reason"] = "fewer_than_minimum_matches"
        return {
            **v5,
            "selector_version": "action-conditioned-v6-development",
            "parent_selector_version": v5.get("selector_version"),
            "action_conditioned_rescue": rescue,
        }
    cv2.setRNGSeed(RNG_SEED)
    matrix, inliers = V1.fit_family(selected_family, source_points, target_points)
    if matrix is None:
        rescue["reason"] = "candidate_refit_failed"
        return {
            **v5,
            "selector_version": "action-conditioned-v6-development",
            "parent_selector_version": v5.get("selector_version"),
            "action_conditioned_rescue": rescue,
        }
    evidence = _measure_candidate(
        source, target, action, selected_family, matrix, inliers, source_points, target_points
    )
    geometric_checks, geometric_pass = _geometric_certificate(
        evidence.geometric, evidence.photometric, evidence.nearest_fraction
    )
    appearance_checks, absolute_appearance, structural_appearance, appearance_pass = _appearance_certificate(
        evidence.appearance_support, evidence.photometric, evidence.nearest_fraction
    )
    accepted = bool(
        all(evidence.fit_checks.values())
        and evidence.stability["pass"]
        and evidence.shape["pass"]
        and evidence.in_bounds
        and (geometric_pass or appearance_pass)
    )
    rescue.update(
        {
            "accepted": accepted,
            "reason": "action_local_geometry_and_or_appearance_certified"
            if accepted
            else "action_conditioned_certificate_failed",
            "candidate_matrix": matrix.tolist(),
            "mapped_action": evidence.mapped_action.tolist(),
            "full_fit": {
                "matches": int(len(source_points)),
                "inliers": int(inliers.sum()),
                "inlier_rate": float(inliers.mean()),
                "inlier_median_residual_px": float(np.median(evidence.inlier_residuals))
                if len(evidence.inlier_residuals)
                else None,
                "inlier_p90_residual_px": float(np.quantile(evidence.inlier_residuals, 0.9))
                if len(evidence.inlier_residuals)
                else None,
                "checks": evidence.fit_checks,
            },
            "nearest_inlier_distance_fraction": evidence.nearest_fraction,
            "geometric_support": evidence.geometric,
            "appearance_support": evidence.appearance_support,
            "stability": evidence.stability,
            "local_photometric": evidence.photometric,
            "local_map_shape": evidence.shape,
            "mapped_action_in_target_image_bounds": evidence.in_bounds,
            "geometric_branch": {"pass": geometric_pass, "checks": geometric_checks},
            "appearance_branch": {
                "pass": appearance_pass,
                "absolute_appearance_pass": absolute_appearance,
                "structural_appearance_pass": structural_appearance,
                "checks": appearance_checks,
            },
        }
    )
    return {
        **v5,
        "selector_version": "action-conditioned-v6-development",
        "parent_selector_version": v5.get("selector_version"),
        "supported": accepted,
        "reason": rescue["reason"],
        "decision_path": "action_conditioned_local_rescue",
        "matrix": matrix.tolist() if accepted else None,
        "candidate_matrix_audit": matrix.tolist(),
        "action_conditioned_rescue": rescue,
    }


def _geometric_certificate(geometric, photometric, nearest_fraction):
    """Certify support around the action without changing global-fit decisions."""
    grayscale_ncc = photometric.get("grayscale_ncc")
    geometric_photo_pass = bool(
        photometric.get("valid")
        and photometric.get("valid_rate", 0.0) >= MINIMUM_LOCAL_VALID_RATE
        and (photometric.get("grayscale_mae", math.inf) <= GEOMETRIC_MAXIMUM_GRAYSCALE_MAE)
        and (grayscale_ncc is not None)
        and (grayscale_ncc >= GEOMETRIC_MINIMUM_GRAYSCALE_NCC)
    )
    geometric_checks = {
        "minimum_local_inliers": geometric["inliers"] >= GEOMETRIC_MINIMUM_INLIERS,
        "minimum_angular_bins": geometric["angular_bins"] >= GEOMETRIC_MINIMUM_ANGULAR_BINS,
        "maximum_local_p90_residual": bool(
            geometric["residual_p90_px"] is not None
            and geometric["residual_p90_px"] <= GEOMETRIC_MAXIMUM_LOCAL_P90_RESIDUAL_PX
        ),
        "maximum_nearest_inlier_fraction": bool(
            nearest_fraction is not None and nearest_fraction <= GEOMETRIC_MAXIMUM_NEAREST_INLIER_FRACTION
        ),
        "loose_local_photometric": geometric_photo_pass,
    }
    geometric_pass = all(geometric_checks.values())
    return (geometric_checks, geometric_pass)


def _appearance_certificate(appearance_support, photometric, nearest_fraction):
    """Evaluate strict absolute or structural local appearance evidence."""
    grayscale_ncc = photometric.get("grayscale_ncc")
    absolute_appearance = bool(
        photometric.get("valid")
        and photometric.get("grayscale_mae", math.inf) <= APPEARANCE_MAXIMUM_GRAYSCALE_MAE
        and (photometric.get("grayscale_p90", math.inf) <= APPEARANCE_MAXIMUM_GRAYSCALE_P90)
        and (photometric.get("grayscale_within_12_rate", 0.0) >= APPEARANCE_MINIMUM_WITHIN_12_RATE)
    )
    gradient_ncc = photometric.get("gradient_ncc")
    structural_appearance = bool(
        photometric.get("valid")
        and grayscale_ncc is not None
        and (grayscale_ncc >= APPEARANCE_MINIMUM_GRAYSCALE_NCC)
        and (gradient_ncc is not None)
        and (gradient_ncc >= APPEARANCE_MINIMUM_GRADIENT_NCC)
    )
    appearance_checks = {
        "minimum_local_inliers": appearance_support["inliers"] >= APPEARANCE_MINIMUM_INLIERS,
        "maximum_nearest_inlier_fraction": bool(
            nearest_fraction is not None and nearest_fraction <= APPEARANCE_MAXIMUM_NEAREST_INLIER_FRACTION
        ),
        "minimum_valid_rate": bool(
            photometric.get("valid") and photometric.get("valid_rate", 0.0) >= MINIMUM_LOCAL_VALID_RATE
        ),
        "appearance_reconstruction": absolute_appearance or structural_appearance,
    }
    appearance_pass = all(appearance_checks.values())
    return (appearance_checks, absolute_appearance, structural_appearance, appearance_pass)


def _measure_candidate(
    source, target, action, selected_family, matrix, inliers, source_points, target_points
):
    """Measure fitted map evidence independently of the acceptance policy."""
    mapped_matches = V1.apply_homography(matrix, source_points)
    residuals = np.linalg.norm(mapped_matches - target_points, axis=1)
    inlier_source = source_points[inliers]
    inlier_residuals = residuals[inliers]
    source_diagonal = float(math.hypot(source.shape[1], source.shape[0]))
    distances = np.linalg.norm(inlier_source - action.reshape(2), axis=1)
    fit_checks = {
        "minimum_inliers": int(inliers.sum()) >= MINIMUM_INLIERS,
        "minimum_inlier_rate": float(inliers.mean()) >= MINIMUM_INLIER_RATE,
        "maximum_inlier_median_residual": bool(
            len(inlier_residuals) and float(np.median(inlier_residuals)) <= MAXIMUM_INLIER_MEDIAN_RESIDUAL_PX
        ),
        "maximum_inlier_p90_residual": bool(
            len(inlier_residuals)
            and float(np.quantile(inlier_residuals, 0.9)) <= MAXIMUM_INLIER_P90_RESIDUAL_PX
        ),
    }
    geometric = local_support(
        inlier_source, inlier_residuals, action, source_diagonal, GEOMETRIC_RADIUS_FRACTION
    )
    appearance_support = local_support(
        inlier_source, inlier_residuals, action, source_diagonal, APPEARANCE_RADIUS_FRACTION
    )
    stability = action_stability(selected_family, matrix, source_points, target_points, action)
    photometric = local_photometric(source, target, matrix, action)
    shape = local_map_shape(matrix, action)
    mapped_action = V1.apply_homography(matrix, action)[0]
    nearest_fraction = float(distances.min() / source_diagonal) if len(distances) else None
    in_bounds = mapped_action_in_bounds(mapped_action, target.shape[1], target.shape[0])
    return CandidateEvidence(
        inlier_residuals=inlier_residuals,
        fit_checks=fit_checks,
        geometric=geometric,
        appearance_support=appearance_support,
        stability=stability,
        photometric=photometric,
        shape=shape,
        mapped_action=mapped_action,
        nearest_fraction=nearest_fraction,
        in_bounds=in_bounds,
    )


@dataclass(frozen=True)
class CandidateEvidence:
    inlier_residuals: np.ndarray
    fit_checks: dict[str, Any]
    geometric: dict[str, Any]
    appearance_support: dict[str, Any]
    stability: dict[str, Any]
    photometric: dict[str, Any]
    shape: dict[str, Any]
    mapped_action: np.ndarray
    nearest_fraction: float | None
    in_bounds: bool


def _preserve_parent(v5, action):
    matrix = np.asarray(v5["matrix"], dtype=np.float64)
    mapped = V1.apply_homography(matrix, action)[0]
    return {
        **v5,
        "selector_version": "action-conditioned-v6-development",
        "parent_selector_version": v5.get("selector_version"),
        "action_conditioned_rescue": {
            "eligible": False,
            "accepted": False,
            "reason": "v5_acceptance_preserved",
            "thresholds": thresholds(),
            "source_action": action[0].tolist(),
            "mapped_action": mapped.tolist(),
        },
    }
