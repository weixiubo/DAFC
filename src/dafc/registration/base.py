#!/usr/bin/env python3
"""Target-free visual-contract registration pilot on real ScreenSpot images.

This pilot asks whether a cheap, model-agnostic compiler can recover a single
global action-frame map from source/deployed screenshots and abstain when no
single global map explains the deployment.  Target boxes are used only after
registration to audit action-transfer error.
"""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


from dafc.paths import PROJECT
SOURCE_CACHE = PROJECT / "tmp" / "c327_screenspot_real_model_replication"
OUT_CACHE = PROJECT / "tmp" / "c357_visual_contract_registration_pilot"
SOURCE_REPORT = (
    PROJECT / "validation" / "c327_screenspot_real_model_replication.json"
)
OUT = PROJECT / "validation" / "c357_visual_contract_registration_pilot.json"
PAD = (17, 19, 23)
FEATURE_MAX_DIM = 1400
GRID_N = 17


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def apply_homography(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    homogeneous = np.column_stack((points, np.ones(len(points))))
    mapped = (matrix @ homogeneous.T).T
    return mapped[:, :2] / mapped[:, 2:3]


def normalized_quad(width: int, height: int) -> np.ndarray:
    return np.asarray(
        [[0.0, 0.0], [width - 1.0, 0.0], [width - 1.0, height - 1.0], [0.0, height - 1.0]],
        dtype=np.float32,
    )


def render_global_contract(
    source: np.ndarray, name: str
) -> tuple[np.ndarray, np.ndarray]:
    height, width = source.shape[:2]
    if name == "identity":
        return source.copy(), np.eye(3, dtype=np.float64)

    canvas_w, canvas_h = 1280, 960
    source_quad = normalized_quad(width, height)
    if name == "centered_letterbox":
        scale = min(1024.0 / width, 1024.0 / height)
        content_w = width * scale
        content_h = height * scale
        tx = (1024.0 - content_w) / 2.0
        ty = (1024.0 - content_h) / 2.0
        matrix = np.asarray(
            [[scale, 0.0, tx], [0.0, scale, ty], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return (
            cv2.warpPerspective(
                source, matrix, (1024, 1024), borderValue=PAD
            ),
            matrix,
        )
    if name == "asymmetric_anisotropic_embed":
        base_scale = 0.78 * min(canvas_w / width, canvas_h / height)
        sx = 1.08 * base_scale
        sy = 0.88 * base_scale
        tx = (canvas_w - sx * width) / 2.0 + 31.0
        ty = (canvas_h - sy * height) / 2.0 - 23.0
        matrix = np.asarray(
            [
                [sx, 0.0, tx],
                [0.0, sy, ty],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    elif name == "rotated_remote_panel":
        scale = 0.72 * min(canvas_w / width, canvas_h / height)
        theta = math.radians(7.0)
        cosine = math.cos(theta) * scale
        sine = math.sin(theta) * scale
        source_cx, source_cy = (width - 1.0) / 2.0, (height - 1.0) / 2.0
        target_cx, target_cy = canvas_w / 2.0 + 24.0, canvas_h / 2.0 - 18.0
        matrix = np.asarray(
            [
                [cosine, -sine, target_cx - cosine * source_cx + sine * source_cy],
                [sine, cosine, target_cy - sine * source_cx - cosine * source_cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
    elif name == "perspective_kiosk":
        scale = 0.76 * min(canvas_w / width, canvas_h / height)
        content_w = width * scale
        content_h = height * scale
        left = (canvas_w - content_w) / 2.0
        top = (canvas_h - content_h) / 2.0
        target_quad = np.asarray(
            [
                [left + 0.035 * content_w, top + 0.015 * content_h],
                [left + 0.985 * content_w, top + 0.055 * content_h],
                [left + 0.945 * content_w, top + 0.975 * content_h],
                [left + 0.015 * content_w, top + 0.935 * content_h],
            ],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(source_quad, target_quad).astype(np.float64)
    else:
        raise ValueError(f"unknown global contract: {name}")
    return (
        cv2.warpPerspective(
            source, matrix, (canvas_w, canvas_h), borderValue=PAD
        ),
        matrix,
    )


def render_unsupported(source: np.ndarray, name: str) -> np.ndarray:
    resized = cv2.resize(source, (1024, 768), interpolation=cv2.INTER_AREA)
    if name == "piecewise_vertical_shift":
        target = np.full((896, 1024, 3), PAD, dtype=np.uint8)
        half = resized.shape[1] // 2
        target[32:800, :half] = resized[:, :half]
        target[96:864, half:] = resized[:, half:]
        return target
    if name == "quadrant_permutation":
        target = resized.copy()
        half_w = target.shape[1] // 2
        half_h = target.shape[0] // 2
        quadrants = [
            resized[:half_h, :half_w].copy(),
            resized[:half_h, half_w:].copy(),
            resized[half_h:, :half_w].copy(),
            resized[half_h:, half_w:].copy(),
        ]
        target[:half_h, :half_w] = quadrants[3]
        target[:half_h, half_w:] = quadrants[2]
        target[half_h:, :half_w] = quadrants[1]
        target[half_h:, half_w:] = quadrants[0]
        return target
    raise ValueError(f"unknown unsupported contract: {name}")


def resize_for_features(image: np.ndarray) -> tuple[np.ndarray, float]:
    height, width = image.shape[:2]
    scale = min(1.0, FEATURE_MAX_DIM / max(height, width))
    if scale == 1.0:
        return image, scale
    return (
        cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        ),
        scale,
    )


def mutual_ratio_matches(
    source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    source_small, source_scale = resize_for_features(source)
    target_small, target_scale = resize_for_features(target)
    detector = cv2.SIFT_create(
        nfeatures=5000,
        contrastThreshold=0.02,
        edgeThreshold=12,
        sigma=1.6,
    )
    source_gray = cv2.cvtColor(source_small, cv2.COLOR_BGR2GRAY)
    target_gray = cv2.cvtColor(target_small, cv2.COLOR_BGR2GRAY)
    source_keypoints, source_descriptors = detector.detectAndCompute(source_gray, None)
    target_keypoints, target_descriptors = detector.detectAndCompute(target_gray, None)
    counts = {
        "source_keypoints": len(source_keypoints),
        "target_keypoints": len(target_keypoints),
        "mutual_ratio_matches": 0,
    }
    if source_descriptors is None or target_descriptors is None:
        return np.empty((0, 2)), np.empty((0, 2)), counts
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    def ratio_map(left: np.ndarray, right: np.ndarray) -> dict[int, int]:
        mapping: dict[int, int] = {}
        for pair in matcher.knnMatch(left, right, k=2):
            if len(pair) == 2 and pair[0].distance < 0.76 * pair[1].distance:
                mapping[pair[0].queryIdx] = pair[0].trainIdx
        return mapping

    forward = ratio_map(source_descriptors, target_descriptors)
    reverse = ratio_map(target_descriptors, source_descriptors)
    pairs = sorted(
        (source_index, target_index)
        for source_index, target_index in forward.items()
        if reverse.get(target_index) == source_index
    )
    counts["mutual_ratio_matches"] = len(pairs)
    source_points = np.asarray(
        [source_keypoints[left].pt for left, _ in pairs], dtype=np.float64
    )
    target_points = np.asarray(
        [target_keypoints[right].pt for _, right in pairs], dtype=np.float64
    )
    if len(pairs):
        source_points /= source_scale
        target_points /= target_scale
    return source_points, target_points, counts


def deterministic_split(
    source_points: np.ndarray, target_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if len(source_points) < 2:
        return source_points, target_points, source_points, target_points
    scores = []
    for index, (source, target) in enumerate(zip(source_points, target_points)):
        key = f"{index}:{source[0]:.3f}:{source[1]:.3f}:{target[0]:.3f}:{target[1]:.3f}"
        scores.append(int(hashlib.sha256(key.encode()).hexdigest(), 16))
    order = np.argsort(np.asarray(scores, dtype=object))
    validation_count = max(8, round(0.25 * len(order)))
    validation_count = min(validation_count, max(1, len(order) - 8))
    validation = order[:validation_count]
    train = order[validation_count:]
    return (
        source_points[train],
        target_points[train],
        source_points[validation],
        target_points[validation],
    )


def fit_family(
    family: str, source: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray]:
    if family == "similarity":
        affine, mask = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
            maxIters=5000,
            confidence=0.999,
        )
        if affine is None:
            return None, np.zeros(len(source), dtype=bool)
        matrix = np.vstack((affine, [0.0, 0.0, 1.0]))
    elif family == "affine":
        affine, mask = cv2.estimateAffine2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
            maxIters=5000,
            confidence=0.999,
        )
        if affine is None:
            return None, np.zeros(len(source), dtype=bool)
        matrix = np.vstack((affine, [0.0, 0.0, 1.0]))
    elif family == "homography":
        matrix, mask = cv2.findHomography(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=4.0,
            maxIters=5000,
            confidence=0.999,
        )
        if matrix is None:
            return None, np.zeros(len(source), dtype=bool)
    else:
        raise ValueError(f"unknown family: {family}")
    return np.asarray(matrix, dtype=np.float64), np.asarray(mask).reshape(-1).astype(bool)


def coverage_fraction(points: np.ndarray, width: int, height: int) -> float:
    if len(points) < 3:
        return 0.0
    hull = cv2.convexHull(np.asarray(points, dtype=np.float32))
    return float(cv2.contourArea(hull)) / float(width * height)


def evaluate_fit(
    matrix: np.ndarray,
    train_source: np.ndarray,
    train_target: np.ndarray,
    train_inliers: np.ndarray,
    validation_source: np.ndarray,
    validation_target: np.ndarray,
    source_width: int,
    source_height: int,
) -> dict[str, Any]:
    train_errors = np.linalg.norm(
        apply_homography(matrix, train_source) - train_target, axis=1
    )
    validation_errors = np.linalg.norm(
        apply_homography(matrix, validation_source) - validation_target, axis=1
    )
    inlier_points = train_source[train_inliers]
    return {
        "train_matches": int(len(train_source)),
        "train_inliers": int(train_inliers.sum()),
        "train_inlier_rate": float(train_inliers.mean()) if len(train_inliers) else 0.0,
        "train_inlier_median_error": float(np.median(train_errors[train_inliers]))
        if train_inliers.any()
        else None,
        "source_coverage": coverage_fraction(
            inlier_points, source_width, source_height
        ),
        "validation_matches": int(len(validation_source)),
        "validation_median_error": float(np.median(validation_errors))
        if len(validation_errors)
        else None,
        "validation_p90_error": float(np.quantile(validation_errors, 0.90))
        if len(validation_errors)
        else None,
        "validation_within_6px_rate": float((validation_errors <= 6.0).mean())
        if len(validation_errors)
        else 0.0,
    }


def fit_supported(metrics: dict[str, Any]) -> bool:
    return (
        metrics["train_inliers"] >= 16
        and metrics["train_inlier_rate"] >= 0.76
        and metrics["train_inlier_median_error"] is not None
        and metrics["train_inlier_median_error"] <= 2.5
        and metrics["source_coverage"] >= 0.30
        and metrics["validation_matches"] >= 8
        and metrics["validation_median_error"] is not None
        and metrics["validation_median_error"] <= 3.0
        and metrics["validation_within_6px_rate"] >= 0.60
    )


def register_contract(source: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    source_points, target_points, counts = mutual_ratio_matches(source, target)
    if len(source_points) < 32:
        return {
            "supported": False,
            "selected_family": None,
            "reason": "fewer_than_32_mutual_ratio_matches",
            "match_counts": counts,
            "families": {},
            "matrix": None,
        }
    train_source, train_target, validation_source, validation_target = (
        deterministic_split(source_points, target_points)
    )
    families = {}
    candidates: list[tuple[float, str, np.ndarray]] = []
    parameter_counts = {"similarity": 4, "affine": 6, "homography": 8}
    for family in ("similarity", "affine", "homography"):
        matrix, inliers = fit_family(family, train_source, train_target)
        if matrix is None:
            families[family] = {"fit": False, "supported": False}
            continue
        metrics = evaluate_fit(
            matrix,
            train_source,
            train_target,
            inliers,
            validation_source,
            validation_target,
            source.shape[1],
            source.shape[0],
        )
        supported = fit_supported(metrics)
        selection_score = (
            float(metrics["train_inlier_median_error"] or 0.0)
            + float(metrics["validation_median_error"] or 0.0)
            + 0.05 * (parameter_counts[family] - 4)
        )
        families[family] = {"fit": True, "supported": supported, **metrics}
        families[family]["selection_score"] = selection_score
        if supported:
            candidates.append((selection_score, family, matrix))
    if candidates:
        _, selected_family, selected_matrix = min(candidates, key=lambda item: item[0])
    else:
        selected_family = None
        selected_matrix = None
    return {
        "supported": selected_family is not None,
        "selected_family": selected_family,
        "reason": "accepted_minimal_global_family"
        if selected_family is not None
        else "no_global_family_passed_target_free_validation",
        "match_counts": counts,
        "families": families,
        "matrix": selected_matrix.tolist() if selected_matrix is not None else None,
    }


def source_grid(width: int, height: int) -> np.ndarray:
    xs = np.linspace(0.04 * width, 0.96 * width, GRID_N)
    ys = np.linspace(0.04 * height, 0.96 * height, GRID_N)
    return np.asarray([(x, y) for y in ys for x in xs], dtype=np.float64)


def bbox_center(annotation: dict[str, Any]) -> np.ndarray:
    x0, y0, x1, y1 = [float(value) for value in annotation["bbox_normalized"]]
    return np.asarray(
        [[
            (x0 + x1) * float(annotation["image_width"]) / 2.0,
            (y0 + y1) * float(annotation["image_height"]) / 2.0,
        ]],
        dtype=np.float64,
    )


def unique_source_rows(report: dict[str, Any], rows: int) -> list[int]:
    selected = []
    seen = set()
    for row in report["rows"]:
        row_idx = int(row["row_idx"])
        if row_idx not in seen:
            seen.add(row_idx)
            selected.append(row_idx)
        if len(selected) == rows:
            return selected
    raise RuntimeError(f"report contains fewer than {rows} unique rows")


def run(rows: int, cache_dir: Path) -> dict[str, Any]:
    source_report = json.loads(SOURCE_REPORT.read_text(encoding="utf-8"))
    selected = unique_source_rows(source_report, rows)
    global_contracts = [
        "identity",
        "centered_letterbox",
        "asymmetric_anisotropic_embed",
        "rotated_remote_panel",
        "perspective_kiosk",
    ]
    unsupported_contracts = [
        "piecewise_vertical_shift",
        "quadrant_permutation",
    ]
    records = []
    for row_idx in selected:
        source_dir = SOURCE_CACHE / "samples" / f"row_{row_idx:04d}"
        source_path = source_dir / "source.png"
        annotation_path = source_dir / "annotation.json"
        source = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
        if source is None:
            raise FileNotFoundError(source_path)
        annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
        if annotation["image_sha256"] != sha256(source_path):
            raise RuntimeError(f"source hash mismatch at row {row_idx}")
        row_dir = cache_dir / f"row_{row_idx:04d}"
        row_dir.mkdir(parents=True, exist_ok=True)

        for contract in global_contracts:
            target, true_matrix = render_global_contract(source, contract)
            target_path = row_dir / f"{contract}.png"
            cv2.imwrite(str(target_path), target)
            registration = register_contract(source, target)
            ground_truth = None
            if registration["matrix"] is not None:
                estimated = np.asarray(registration["matrix"], dtype=np.float64)
                grid = source_grid(source.shape[1], source.shape[0])
                errors = np.linalg.norm(
                    apply_homography(estimated, grid)
                    - apply_homography(true_matrix, grid),
                    axis=1,
                )
                center = bbox_center(annotation)
                target_center = apply_homography(true_matrix, center)[0]
                estimated_center = apply_homography(estimated, center)[0]
                ground_truth = {
                    "grid_median_error": float(np.median(errors)),
                    "grid_p95_error": float(np.quantile(errors, 0.95)),
                    "grid_max_error": float(errors.max()),
                    "annotation_center_error": float(
                        np.linalg.norm(estimated_center - target_center)
                    ),
                }
            records.append(
                {
                    "row_idx": row_idx,
                    "declared_support": True,
                    "contract": contract,
                    "source_sha256": sha256(source_path),
                    "target_sha256": sha256(target_path),
                    "registration": registration,
                    "ground_truth_audit": ground_truth,
                }
            )

        for contract in unsupported_contracts:
            target = render_unsupported(source, contract)
            target_path = row_dir / f"{contract}.png"
            cv2.imwrite(str(target_path), target)
            registration = register_contract(source, target)
            records.append(
                {
                    "row_idx": row_idx,
                    "declared_support": False,
                    "contract": contract,
                    "source_sha256": sha256(source_path),
                    "target_sha256": sha256(target_path),
                    "registration": registration,
                    "ground_truth_audit": None,
                }
            )

    supported = [record for record in records if record["declared_support"]]
    unsupported = [record for record in records if not record["declared_support"]]
    supported_accepted = [record for record in supported if record["registration"]["supported"]]
    unsupported_rejected = [record for record in unsupported if not record["registration"]["supported"]]
    family_confusion = Counter(
        (record["contract"], record["registration"]["selected_family"] or "abstain")
        for record in records
    )
    errors = [
        record["ground_truth_audit"]["grid_p95_error"]
        for record in supported_accepted
        if record["ground_truth_audit"] is not None
    ]
    per_contract: dict[str, dict[str, Any]] = {}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[record["contract"]].append(record)
    for contract, group in sorted(grouped.items()):
        accepted = sum(int(row["registration"]["supported"]) for row in group)
        group_errors = [
            row["ground_truth_audit"]["grid_p95_error"]
            for row in group
            if row["ground_truth_audit"] is not None
            and row["registration"]["supported"]
        ]
        per_contract[contract] = {
            "declared_support": bool(group[0]["declared_support"]),
            "accepted": accepted,
            "total": len(group),
            "accepted_rate": accepted / len(group),
            "median_grid_p95_error": float(np.median(group_errors))
            if group_errors
            else None,
            "max_grid_p95_error": float(max(group_errors))
            if group_errors
            else None,
        }
    criteria = {
        "supported_accept_rate_at_least_95": len(supported_accepted)
        / len(supported)
        >= 0.95,
        "unsupported_reject_rate_at_least_95": len(unsupported_rejected)
        / len(unsupported)
        >= 0.95,
        "accepted_supported_grid_p95_below_3px": bool(errors)
        and max(errors) < 3.0,
        "identity_all_supported": per_contract["identity"]["accepted"] == rows,
        "identity_grid_p95_below_0p5px": (
            per_contract["identity"]["max_grid_p95_error"] is not None
            and per_contract["identity"]["max_grid_p95_error"] < 0.5
        ),
    }
    return {
        "candidate": "C357-visual-contract-registration-pilot",
        "scope": "model-free target-free feasibility pilot; not paper evidence",
        "source_report": SOURCE_REPORT.relative_to(PROJECT).as_posix(),
        "source_report_sha256": sha256(SOURCE_REPORT),
        "selected_row_indices": selected,
        "rows": rows,
        "global_contracts": global_contracts,
        "unsupported_contracts": unsupported_contracts,
        "summary": {
            "supported_accepted": len(supported_accepted),
            "supported_total": len(supported),
            "supported_accept_rate": len(supported_accepted) / len(supported),
            "unsupported_rejected": len(unsupported_rejected),
            "unsupported_total": len(unsupported),
            "unsupported_reject_rate": len(unsupported_rejected) / len(unsupported),
            "family_confusion": {
                f"{contract}::{family}": count
                for (contract, family), count in sorted(family_confusion.items())
            },
            "per_contract": per_contract,
        },
        "criteria": criteria,
        "pass": all(criteria.values()),
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--cache-dir", default=str(OUT_CACHE))
    parser.add_argument("--output", default=str(OUT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args.rows, Path(args.cache_dir).resolve())
    json_write(Path(args.output).resolve(), report)
    print(json.dumps({"summary": report["summary"], "criteria": report["criteria"], "pass": report["pass"]}, indent=2))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
