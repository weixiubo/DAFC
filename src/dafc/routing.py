from __future__ import annotations
import hashlib, json, math, time
from pathlib import Path
from typing import Any
import cv2
import numpy as np
from dafc.evidence import sha256
from dafc.registration import selector_v5 as V5, action_contract as V6


POLICIES = ("always", "s_only", "standard_ransac", "dafc")

STANDARD_RNG_SEED = 585001

STANDARD_RANSAC_THRESHOLD_PX = 4.0

STANDARD_MAX_ITERS = 5_000

STANDARD_CONFIDENCE = 0.999

STANDARD_MIN_INLIERS = 16

STANDARD_MIN_INLIER_RATIO = 0.90

TARGET_BOUNDARY_MARGIN_PX = 4.0

def apply_homography(matrix: np.ndarray, point: list[float]) -> list[float] | None:
    mapped = cv2.perspectiveTransform(
        np.asarray(point, dtype=np.float64).reshape(1, 1, 2),
        np.asarray(matrix, dtype=np.float64),
    ).reshape(2)
    if not np.isfinite(mapped).all():
        return None
    return [float(mapped[0]), float(mapped[1])]

def point_in_bounds(point: list[float] | None, image: np.ndarray, margin: float = 0.0) -> bool:
    if point is None or len(point) != 2 or not all(math.isfinite(v) for v in point):
        return False
    height, width = image.shape[:2]
    return bool(
        margin <= point[0] < width - margin
        and margin <= point[1] < height - margin
    )

def standard_registration(
    source: np.ndarray, target: np.ndarray, action_xy: list[float]
) -> dict[str, Any]:
    started = time.perf_counter()
    source_points, target_points, counts = V5.v1.mutual_ratio_matches(source, target)
    if len(source_points) < 4:
        return {
            "accepted": False,
            "reason": "fewer_than_four_correspondences",
            "match_counts": counts,
            "matches": int(len(source_points)),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "matrix": None,
            "mapped_action_xy": None,
            "runtime_s": time.perf_counter() - started,
        }
    cv2.setRNGSeed(STANDARD_RNG_SEED)
    matrix, mask = cv2.findHomography(
        source_points.astype(np.float64),
        target_points.astype(np.float64),
        method=cv2.RANSAC,
        ransacReprojThreshold=STANDARD_RANSAC_THRESHOLD_PX,
        maxIters=STANDARD_MAX_ITERS,
        confidence=STANDARD_CONFIDENCE,
    )
    if matrix is None or mask is None or not np.isfinite(matrix).all():
        return {
            "accepted": False,
            "reason": "homography_fit_failed",
            "match_counts": counts,
            "matches": int(len(source_points)),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "matrix": None,
            "mapped_action_xy": None,
            "runtime_s": time.perf_counter() - started,
        }
    inliers = int(mask.reshape(-1).astype(bool).sum())
    ratio = inliers / len(source_points)
    mapped = apply_homography(matrix, action_xy)
    accepted = bool(
        inliers >= STANDARD_MIN_INLIERS
        and ratio >= STANDARD_MIN_INLIER_RATIO
        and point_in_bounds(mapped, target, TARGET_BOUNDARY_MARGIN_PX)
    )
    reasons = []
    if inliers < STANDARD_MIN_INLIERS:
        reasons.append("too_few_inliers")
    if ratio < STANDARD_MIN_INLIER_RATIO:
        reasons.append("inlier_ratio_below_0.90")
    if not point_in_bounds(mapped, target, TARGET_BOUNDARY_MARGIN_PX):
        reasons.append("mapped_action_out_of_bounds")
    return {
        "accepted": accepted,
        "reason": "accepted" if accepted else "+".join(reasons),
        "match_counts": counts,
        "matches": int(len(source_points)),
        "inliers": inliers,
        "inlier_ratio": float(ratio),
        "matrix": matrix.tolist(),
        "mapped_action_xy": mapped,
        "runtime_s": time.perf_counter() - started,
    }

def compact_dafc(receipt: dict[str, Any], action_xy: list[float]) -> dict[str, Any]:
    mapped = None
    if receipt.get("supported") and receipt.get("matrix") is not None:
        mapped = apply_homography(np.asarray(receipt["matrix"], dtype=np.float64), action_xy)
    rescue = receipt.get("action_conditioned_rescue") or {}
    return {
        "accepted": bool(receipt.get("supported") and mapped is not None),
        "reason": receipt.get("reason"),
        "decision_path": receipt.get("decision_path"),
        "selected_family": receipt.get("selected_family"),
        "selector_version": receipt.get("selector_version"),
        "mapped_action_xy": mapped,
        "matrix": receipt.get("matrix"),
        "action_conditioned_rescue": {
            "eligible": rescue.get("eligible"),
            "accepted": rescue.get("accepted"),
            "reason": rescue.get("reason"),
        },
    }

def authorize(
    policy: str,
    source: dict[str, Any],
    target: np.ndarray,
) -> dict[str, Any]:
    if policy not in POLICIES or policy == "always":
        return {
            "accepted": False,
            "reason": "always_reground",
            "mapped_action_xy": None,
            "runtime_s": 0.0,
        }
    if not source["source_success"]:
        return {
            "accepted": False,
            "reason": "source_receipt_failed",
            "mapped_action_xy": None,
            "runtime_s": 0.0,
        }
    memory = source["atomic_memory"]
    source_path = Path(memory["source_before"])
    if sha256(source_path) != memory["source_before_sha256"]:
        raise RuntimeError("source before image changed at authorization")
    source_image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
    if source_image is None:
        raise RuntimeError("could not decode source before image")
    action = [float(value) for value in memory["source_action_xy"]]
    if policy == "s_only":
        return {
            "accepted": True,
            "reason": "source_receipt_only_raw_coordinate",
            "mapped_action_xy": action,
            "runtime_s": 0.0,
        }
    if policy == "standard_ransac":
        return standard_registration(source_image, target, action)
    started = time.perf_counter()
    receipt = V6.register_action_contract(source_image, target, action)
    compact = compact_dafc(receipt, action)
    compact["runtime_s"] = time.perf_counter() - started
    if compact["accepted"] and not point_in_bounds(
        compact["mapped_action_xy"], target, TARGET_BOUNDARY_MARGIN_PX
    ):
        raise RuntimeError("DAFC accepted an out-of-bounds action")
    return compact
