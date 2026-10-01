#!/usr/bin/env python3
"""Safety-first v2 target-free visual action-contract selector.

This module is a post-C358 development revision.  It must be validated on a
new disjoint image set before any paper-facing use.
"""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import hashlib
from typing import Any

import cv2
import numpy as np

import dafc.registration.base as v1


PARAMETER_COUNTS = {"similarity": 4, "affine": 6, "homography": 8}
STABILITY_REPETITIONS = 5
STABILITY_KEEP_FRACTION = 0.80
MAX_STABILITY_GRID_P95_PX = 3.0


def selection_score(family: str, metrics: dict[str, Any]) -> float:
    return (
        float(metrics["train_inlier_median_error"] or 0.0)
        + float(metrics["validation_median_error"] or 0.0)
        + 0.05 * (PARAMETER_COUNTS[family] - 4)
    )


def deterministic_subsample_mask(
    source_points: np.ndarray,
    target_points: np.ndarray,
    repetition: int,
) -> np.ndarray:
    keep = []
    cutoff = round(STABILITY_KEEP_FRACTION * 10_000)
    for index, (source, target) in enumerate(zip(source_points, target_points)):
        key = (
            f"{repetition}:{index}:{source[0]:.3f}:{source[1]:.3f}:"
            f"{target[0]:.3f}:{target[1]:.3f}"
        )
        value = int(hashlib.sha256(key.encode()).hexdigest(), 16) % 10_000
        keep.append(value < cutoff)
    return np.asarray(keep, dtype=bool)


def fit_stability(
    family: str,
    final_matrix: np.ndarray,
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_width: int,
    source_height: int,
) -> dict[str, Any]:
    grid = v1.source_grid(source_width, source_height)
    final_grid = v1.apply_homography(final_matrix, grid)
    repetitions = []
    for repetition in range(STABILITY_REPETITIONS):
        mask = deterministic_subsample_mask(
            source_points, target_points, repetition
        )
        subset_source = source_points[mask]
        subset_target = target_points[mask]
        if len(subset_source) < 32:
            repetitions.append(
                {
                    "repetition": repetition,
                    "fit": False,
                    "reason": "fewer_than_32_subsample_matches",
                }
            )
            continue
        matrix, inliers = v1.fit_family(family, subset_source, subset_target)
        if matrix is None or int(inliers.sum()) < 16:
            repetitions.append(
                {
                    "repetition": repetition,
                    "fit": False,
                    "reason": "subsample_fit_or_inlier_failure",
                }
            )
            continue
        disagreement = np.linalg.norm(
            v1.apply_homography(matrix, grid) - final_grid,
            axis=1,
        )
        repetitions.append(
            {
                "repetition": repetition,
                "fit": True,
                "matches": int(len(subset_source)),
                "inliers": int(inliers.sum()),
                "inlier_rate": float(inliers.mean()),
                "source_coverage": v1.coverage_fraction(
                    subset_source[inliers], source_width, source_height
                ),
                "grid_disagreement_median": float(np.median(disagreement)),
                "grid_disagreement_p95": float(
                    np.quantile(disagreement, 0.95)
                ),
            }
        )
    fitted = [item for item in repetitions if item.get("fit")]
    maximum_p95 = (
        max(item["grid_disagreement_p95"] for item in fitted)
        if fitted
        else None
    )
    passed = (
        len(fitted) == STABILITY_REPETITIONS
        and maximum_p95 is not None
        and maximum_p95 <= MAX_STABILITY_GRID_P95_PX
    )
    return {
        "repetitions": STABILITY_REPETITIONS,
        "keep_fraction": STABILITY_KEEP_FRACTION,
        "maximum_grid_disagreement_p95": maximum_p95,
        "threshold_px": MAX_STABILITY_GRID_P95_PX,
        "pass": passed,
        "details": repetitions,
    }


def exact_identity(source: np.ndarray, target: np.ndarray) -> bool:
    return source.shape == target.shape and np.array_equal(source, target)


def register_contract(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    if exact_identity(source, target):
        return {
            "selector_version": "safety-first-v2",
            "supported": True,
            "selected_family": "identity",
            "reason": "exact_pixel_identity",
            "match_counts": {
                "source_keypoints": 0,
                "target_keypoints": 0,
                "mutual_ratio_matches": 0,
            },
            "families": {},
            "selected_family_support": {
                "exact_pixel_identity": True,
            },
            "full_refit": {
                "matches": 0,
                "inliers": 0,
                "inlier_rate": 1.0,
                "source_coverage": 1.0,
            },
            "stability": {
                "repetitions": 0,
                "maximum_grid_disagreement_p95": 0.0,
                "threshold_px": MAX_STABILITY_GRID_P95_PX,
                "pass": True,
                "details": [],
            },
            "matrix": np.eye(3, dtype=np.float64).tolist(),
        }

    source_points, target_points, counts = v1.mutual_ratio_matches(
        source, target
    )
    if len(source_points) < 32:
        return {
            "selector_version": "safety-first-v2",
            "supported": False,
            "selected_family": None,
            "reason": "fewer_than_32_mutual_ratio_matches",
            "match_counts": counts,
            "families": {},
            "selected_family_support": None,
            "full_refit": None,
            "stability": None,
            "matrix": None,
        }
    train_source, train_target, validation_source, validation_target = (
        v1.deterministic_split(source_points, target_points)
    )
    families: dict[str, dict[str, Any]] = {}
    fitted: list[tuple[float, str, np.ndarray, dict[str, Any]]] = []
    for family in ("similarity", "affine", "homography"):
        matrix, inliers = v1.fit_family(family, train_source, train_target)
        if matrix is None:
            families[family] = {"fit": False, "supported": False}
            continue
        metrics = v1.evaluate_fit(
            matrix,
            train_source,
            train_target,
            inliers,
            validation_source,
            validation_target,
            source.shape[1],
            source.shape[0],
        )
        supported = v1.fit_supported(metrics)
        score = selection_score(family, metrics)
        families[family] = {
            "fit": True,
            "supported": supported,
            "selection_score": score,
            **metrics,
        }
        fitted.append((score, family, matrix, metrics))
    if not fitted:
        return {
            "selector_version": "safety-first-v2",
            "supported": False,
            "selected_family": None,
            "reason": "no_family_fit",
            "match_counts": counts,
            "families": families,
            "selected_family_support": None,
            "full_refit": None,
            "stability": None,
            "matrix": None,
        }

    _, selected_family, _, selected_metrics = min(
        fitted, key=lambda item: item[0]
    )
    if not v1.fit_supported(selected_metrics):
        return {
            "selector_version": "safety-first-v2",
            "supported": False,
            "selected_family": selected_family,
            "reason": "lowest_score_family_lacks_global_support",
            "match_counts": counts,
            "families": families,
            "selected_family_support": selected_metrics,
            "full_refit": None,
            "stability": None,
            "matrix": None,
        }

    final_matrix, final_inliers = v1.fit_family(
        selected_family, source_points, target_points
    )
    if final_matrix is None:
        return {
            "selector_version": "safety-first-v2",
            "supported": False,
            "selected_family": selected_family,
            "reason": "full_refit_failed",
            "match_counts": counts,
            "families": families,
            "selected_family_support": selected_metrics,
            "full_refit": None,
            "stability": None,
            "matrix": None,
        }
    full_refit = {
        "matches": int(len(source_points)),
        "inliers": int(final_inliers.sum()),
        "inlier_rate": float(final_inliers.mean()),
        "source_coverage": v1.coverage_fraction(
            source_points[final_inliers], source.shape[1], source.shape[0]
        ),
    }
    stability = fit_stability(
        selected_family,
        final_matrix,
        source_points,
        target_points,
        source.shape[1],
        source.shape[0],
    )
    if not stability["pass"]:
        return {
            "selector_version": "safety-first-v2",
            "supported": False,
            "selected_family": selected_family,
            "reason": "subsample_mapping_instability",
            "match_counts": counts,
            "families": families,
            "selected_family_support": selected_metrics,
            "full_refit": full_refit,
            "stability": stability,
            "matrix": None,
        }
    return {
        "selector_version": "safety-first-v2",
        "supported": True,
        "selected_family": selected_family,
        "reason": "lowest_score_family_supported_and_stable",
        "match_counts": counts,
        "families": families,
        "selected_family_support": selected_metrics,
        "full_refit": full_refit,
        "stability": stability,
        "matrix": final_matrix.tolist(),
    }


def main() -> int:
    print(
        {
            "selector_version": "safety-first-v2",
            "status": "development_only_requires_disjoint_validation",
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
