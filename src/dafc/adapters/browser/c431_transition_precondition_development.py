#!/usr/bin/env python3
"""Develop an observable-effect precondition for visual transition memory.

The input is the C430 DOM-oracle mechanics audit.  This script calls no GUI
model and produces no confirmatory paper evidence.  It tests one fixed,
interpretable rule: a transition is applicable only when (1) its source
pre/post effect is observable, (2) the aligned query is closer to the stored
pre-state than the post-state on effect pixels, and (3) nearby stable context
still agrees.  Known image transforms stand in for a successful geometry gate
so that precondition development is isolated from registration development.
"""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import json
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np


from dafc.paths import PROJECT
INPUT = PROJECT / "validation" / "c430_multistep_transition_feasibility.json"
OUTPUT = PROJECT / "validation" / "c431_transition_precondition_development.json"

# Development constants.  They are intentionally simple and will be frozen in
# a prospective protocol before any fresh model trajectory is evaluated.
PIXEL_EFFECT_THRESHOLD = 12.0
MIN_EFFECT_PIXELS = 16
MIN_EFFECT_SEPARATION = 8.0
MIN_NORMALIZED_MARGIN = 0.20
MAX_CONTEXT_ERROR = 0.12

VARIANTS: tuple[dict[str, Any], ...] = (
    {"name": "identity", "geometry": "identity", "jpeg_quality": None},
    {"name": "jpeg_82", "geometry": "identity", "jpeg_quality": 82},
    {"name": "zoom_125_jpeg_82", "geometry": "zoom_125", "jpeg_quality": 82},
    {"name": "zoom_80_jpeg_82", "geometry": "zoom_80", "jpeg_quality": 82},
    {"name": "affine_jpeg_82", "geometry": "affine", "jpeg_quality": 82},
)


@lru_cache(maxsize=None)
def load_image(relative_path: str) -> np.ndarray:
    image = cv2.imread(str(PROJECT / relative_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read {relative_path}; rerun C430 first")
    return image


def apply_geometry(image: np.ndarray, geometry: str) -> np.ndarray:
    height, width = image.shape[:2]
    if geometry == "identity":
        return image.copy()
    if geometry == "zoom_125":
        return cv2.resize(
            image,
            (round(width * 1.25), round(height * 1.25)),
            interpolation=cv2.INTER_LINEAR,
        )
    if geometry == "zoom_80":
        return cv2.resize(
            image,
            (round(width * 0.80), round(height * 0.80)),
            interpolation=cv2.INTER_AREA,
        )
    if geometry == "affine":
        # Pixel-space analogue of the development browser affine contract.
        matrix = np.asarray(
            [[1.02, -0.05, 54.0], [0.08, 0.96, 24.0]], dtype=np.float32
        )
        return cv2.warpAffine(
            image,
            matrix,
            (width + 54, height + 48),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
    raise KeyError(geometry)


def jpeg_roundtrip(image: np.ndarray, quality: int | None) -> np.ndarray:
    if quality is None:
        return image
    ok, payload = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed")
    return decoded


def render_template(image: np.ndarray, variant: dict[str, Any]) -> np.ndarray:
    """Warp a stored state with the geometry map accepted by gate G."""

    return apply_geometry(image, str(variant["geometry"]))


def render_query(image: np.ndarray, variant: dict[str, Any]) -> np.ndarray:
    """Render a target observation, including target-stream compression."""

    warped = apply_geometry(image, str(variant["geometry"]))
    return jpeg_roundtrip(warped, variant["jpeg_quality"])


def effect_model(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    before_f = before.astype(np.float32)
    after_f = after.astype(np.float32)
    channel_delta = np.abs(after_f - before_f)
    delta = np.max(channel_delta, axis=2)
    core = delta > PIXEL_EFFECT_THRESHOLD
    core_pixels = int(core.sum())

    # Difference magnitude weights the diagnostic pixels.  This is equivalent
    # to a normalized signed template comparison and gives +1 on the exact
    # pre-state and -1 on the exact post-state.
    weights = np.where(core, delta, 0.0).astype(np.float32)
    weight_sum = float(weights.sum())
    source_pixel_error = np.mean(channel_delta, axis=2)
    separation = (
        float((weights * source_pixel_error).sum() / weight_sum)
        if weight_sum > 0.0
        else 0.0
    )

    core_u8 = core.astype(np.uint8)
    outer = cv2.dilate(core_u8, np.ones((13, 13), np.uint8), iterations=1) > 0
    inner = cv2.dilate(core_u8, np.ones((3, 3), np.uint8), iterations=1) > 0
    context = outer & ~inner
    observable = bool(
        core_pixels >= MIN_EFFECT_PIXELS and separation >= MIN_EFFECT_SEPARATION
    )
    return {
        "weights": weights,
        "context": context,
        "core_pixels": core_pixels,
        "source_separation": separation,
        "observable": observable,
    }


def score_query(
    source_before: np.ndarray,
    source_after: np.ndarray,
    query: np.ndarray,
    model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if source_before.shape != source_after.shape or query.shape != source_before.shape:
        raise RuntimeError(
            f"shape mismatch: {source_before.shape} {source_after.shape} {query.shape}"
        )
    model = model or effect_model(source_before, source_after)
    if not model["observable"]:
        return {
            "observable": False,
            "accepted": False,
            "reason": "unobservable_source_effect",
            "core_pixels": model["core_pixels"],
            "source_separation": model["source_separation"],
            "pre_error": None,
            "post_error": None,
            "normalized_margin": None,
            "context_error": None,
        }

    before_f = source_before.astype(np.float32)
    after_f = source_after.astype(np.float32)
    query_f = query.astype(np.float32)
    weights = model["weights"]
    weight_sum = float(weights.sum())
    pre_pixel_error = np.mean(np.abs(query_f - before_f), axis=2)
    post_pixel_error = np.mean(np.abs(query_f - after_f), axis=2)
    pre_error = float((weights * pre_pixel_error).sum() / weight_sum)
    post_error = float((weights * post_pixel_error).sum() / weight_sum)
    normalized_margin = float(
        (post_error - pre_error) / max(float(model["source_separation"]), 1e-6)
    )

    context = model["context"]
    if int(context.sum()) >= 16:
        context_error = float(
            np.mean(np.abs(query_f[context] - before_f[context])) / 255.0
        )
    else:
        context_error = 0.0
    margin_ok = normalized_margin >= MIN_NORMALIZED_MARGIN
    context_ok = context_error <= MAX_CONTEXT_ERROR
    accepted = bool(margin_ok and context_ok)
    if margin_ok and not context_ok:
        reason = "context_mismatch"
    elif margin_ok:
        reason = "applicable_pre_state"
    elif normalized_margin <= -MIN_NORMALIZED_MARGIN and context_ok:
        reason = "confident_post_state"
    elif normalized_margin <= -MIN_NORMALIZED_MARGIN:
        reason = "post_state_context_mismatch"
    else:
        reason = "ambiguous_state"
    return {
        "observable": True,
        "accepted": accepted,
        "reason": reason,
        "core_pixels": model["core_pixels"],
        "source_separation": model["source_separation"],
        "pre_error": pre_error,
        "post_error": post_error,
        "normalized_margin": normalized_margin,
        "context_error": context_error,
    }


def score_case(
    rendered_before: np.ndarray,
    rendered_after: np.ndarray,
    query: np.ndarray,
    variant: dict[str, Any],
    model: dict[str, Any],
) -> dict[str, Any]:
    rendered_query = render_query(query, variant)
    return score_query(rendered_before, rendered_after, rendered_query, model)


def expected_correct(row: dict[str, Any]) -> bool:
    if row["expected"] == "accept":
        return bool(row["score"]["accepted"])
    if row["expected"] == "reject":
        return not bool(row["score"]["accepted"])
    if row["expected"] == "abstain_unobservable":
        return not bool(row["score"]["observable"])
    raise KeyError(row["expected"])


def aggregate(rows: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    result = []
    for name in sorted(groups):
        group = groups[name]
        observable = [row for row in group if row["score"]["observable"]]
        margins = [
            float(row["score"]["normalized_margin"])
            for row in observable
            if row["score"]["normalized_margin"] is not None
        ]
        result.append(
            {
                key: name,
                "cases": len(group),
                "correct": sum(int(row["correct"]) for row in group),
                "accuracy": float(np.mean([row["correct"] for row in group])),
                "observable_cases": len(observable),
                "accepted_cases": sum(
                    int(row["score"]["accepted"]) for row in group
                ),
                "margin_min": min(margins) if margins else None,
                "margin_median": float(np.median(margins)) if margins else None,
                "margin_max": max(margins) if margins else None,
            }
        )
    return result


def main() -> int:
    source = json.loads(INPUT.read_text(encoding="utf-8"))
    if source.get("paper_evidence") is not False or not source.get("pass"):
        raise RuntimeError("C430 must be a passing development-only record")

    cases: list[dict[str, Any]] = []
    for episode in source["rows"]:
        transitions = episode["transitions"]
        for candidate_index, transition in enumerate(transitions[:-1]):
            before = load_image(transition["before"]["path"])
            after = load_image(transition["after"]["path"])
            canonical_model = effect_model(before, after)
            observable = bool(canonical_model["observable"])
            for variant in VARIANTS:
                rendered_before = render_template(before, variant)
                rendered_after = render_template(after, variant)
                rendered_model = effect_model(rendered_before, rendered_after)
                base = {
                    "task": episode["task"],
                    "seed": episode["seed"],
                    "candidate_step": candidate_index,
                    "candidate_label": transition["descriptor"]["label"],
                    "variant": variant["name"],
                }

                paired_queries = (
                    (
                        "paired_pre",
                        before,
                        "accept" if observable else "abstain_unobservable",
                    ),
                    (
                        "paired_post",
                        after,
                        "reject" if observable else "abstain_unobservable",
                    ),
                    (
                        "midpoint_ambiguous",
                        np.rint(
                            0.5 * before.astype(np.float32)
                            + 0.5 * after.astype(np.float32)
                        ).astype(np.uint8),
                        "reject" if observable else "abstain_unobservable",
                    ),
                )
                for query_class, query, expected in paired_queries:
                    case = {
                        **base,
                        "query_class": query_class,
                        "query_step": candidate_index,
                        "expected": expected,
                        "score": score_case(
                            rendered_before,
                            rendered_after,
                            query,
                            variant,
                            rendered_model,
                        ),
                    }
                    case["correct"] = expected_correct(case)
                    cases.append(case)

                # Once a visible transition has occurred, all later captured
                # pre-action states should invalidate that same memory entry.
                for query_index in range(candidate_index + 1, len(transitions)):
                    query = load_image(transitions[query_index]["before"]["path"])
                    expected = "reject" if observable else "abstain_unobservable"
                    case = {
                        **base,
                        "query_class": "temporally_stale",
                        "query_step": query_index,
                        "expected": expected,
                        "score": score_case(
                            rendered_before,
                            rendered_after,
                            query,
                            variant,
                            rendered_model,
                        ),
                    }
                    case["correct"] = expected_correct(case)
                    cases.append(case)

                # Checkbox order is intentionally free: a later required box
                # is already applicable in an earlier state.  This prevents
                # conflating visual applicability with exact trajectory index.
                if transition["descriptor"]["kind"] == "checkbox_text":
                    for query_index in range(candidate_index):
                        query = load_image(
                            transitions[query_index]["before"]["path"]
                        )
                        expected = (
                            "accept" if observable else "abstain_unobservable"
                        )
                        case = {
                            **base,
                            "query_class": "order_free_applicable",
                            "query_step": query_index,
                            "expected": expected,
                            "score": score_case(
                                rendered_before,
                                rendered_after,
                                query,
                                variant,
                                rendered_model,
                            ),
                        }
                        case["correct"] = expected_correct(case)
                        cases.append(case)

    by_query_class = aggregate(cases, "query_class")
    by_variant = aggregate(cases, "variant")
    query_lookup = {row["query_class"]: row for row in by_query_class}
    criteria = {
        "development_only": True,
        "zero_model_calls": True,
        "known_geometry_only": True,
        "exact_five_variants": len(VARIANTS) == 5,
        "paired_pre_accuracy_at_least_99pct": query_lookup["paired_pre"][
            "accuracy"
        ]
        >= 0.99,
        "paired_post_accuracy_at_least_99pct": query_lookup["paired_post"][
            "accuracy"
        ]
        >= 0.99,
        "midpoint_rejection_at_least_99pct": query_lookup[
            "midpoint_ambiguous"
        ]["accuracy"]
        >= 0.99,
        "temporally_stale_rejection_at_least_95pct": query_lookup[
            "temporally_stale"
        ]["accuracy"]
        >= 0.95,
        "order_free_applicability_at_least_90pct": query_lookup[
            "order_free_applicable"
        ]["accuracy"]
        >= 0.90,
    }
    result = {
        "title": "C431 Observable-Effect Transition Precondition Development",
        "paper_evidence": False,
        "purpose": (
            "develop P independently of G using C430 trajectories and known "
            "image transforms"
        ),
        "input": INPUT.relative_to(PROJECT).as_posix(),
        "constants": {
            "pixel_effect_threshold": PIXEL_EFFECT_THRESHOLD,
            "min_effect_pixels": MIN_EFFECT_PIXELS,
            "min_effect_separation": MIN_EFFECT_SEPARATION,
            "min_normalized_margin": MIN_NORMALIZED_MARGIN,
            "max_context_error": MAX_CONTEXT_ERROR,
        },
        "variants": list(VARIANTS),
        "summary": {
            "cases": len(cases),
            "correct": sum(int(row["correct"]) for row in cases),
            "accuracy": float(np.mean([row["correct"] for row in cases])),
            "accepted": sum(int(row["score"]["accepted"]) for row in cases),
            "observable": sum(int(row["score"]["observable"]) for row in cases),
        },
        "by_query_class": by_query_class,
        "by_variant": by_variant,
        "criteria": criteria,
        "pass": all(criteria.values()),
        "cases": cases,
    }
    OUTPUT.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "summary": result["summary"],
                "by_query_class": by_query_class,
                "by_variant": by_variant,
                "criteria": criteria,
                "pass": result["pass"],
            },
            indent=2,
        )
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
