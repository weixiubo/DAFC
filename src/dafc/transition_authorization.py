"""TransitionGuard: runtime-verified replay of successful GUI transitions.

The module intentionally consumes only source transition pixels, a source
point action, a source success receipt, and the current screenshot.  Goal
retrieval happens outside this module.  Target DOM, target action labels,
environment rewards, and current trajectory step identity are not inputs.

Each successful transition is interpreted as a visual contract:

* S: the source episode has a verified successful outcome;
* G: an action-conditioned visual map reliably transports the source action;
* P_pre: the current screenshot still exhibits the stored pre-action effect;
* P_post: the next screenshot exhibits the stored post-action effect.

Direct replay requires S and G and P_pre.  A failed P_post invalidates the
transition and requires the caller to invoke its unchanged base agent on the
next observation.  An unconfirmed nonterminal transition may be retired from
the active trajectory only when the current screenshot confidently matches
its stored post-state under the same aligned visual-effect predicate.  This
recovers progress made by a base-agent prefix without receiving a target step.
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

import dafc.preconditions as PRECONDITION
import dafc.registration.action_contract as GEOMETRY
from dafc.transition_types import METHOD_VERSION, AuthorizationContext, MatchProvider, Transition


def compact_geometry(receipt: dict[str, Any]) -> dict[str, Any]:
    rescue = receipt.get("action_conditioned_rescue") or {}
    stability = rescue.get("stability") or {}
    return {
        "supported": bool(receipt.get("supported")),
        "reason": receipt.get("reason"),
        "decision_path": receipt.get("decision_path"),
        "selected_family": receipt.get("selected_family"),
        "selector_version": receipt.get("selector_version"),
        "matrix": receipt.get("matrix"),
        "correspondence_frontend": receipt.get("correspondence_frontend"),
        "learned_consensus_rescue": receipt.get("learned_consensus_rescue"),
        "action_conditioned_rescue": {
            "eligible": rescue.get("eligible"),
            "accepted": rescue.get("accepted"),
            "reason": rescue.get("reason"),
            "mapped_action": rescue.get("mapped_action"),
            "full_fit": rescue.get("full_fit"),
            "nearest_inlier_distance_fraction": rescue.get("nearest_inlier_distance_fraction"),
            "geometric_branch": rescue.get("geometric_branch"),
            "appearance_branch": rescue.get("appearance_branch"),
            "stability": {
                "pass": stability.get("pass"),
                "maximum_disagreement_px": stability.get("maximum_disagreement_px"),
                "fitted_repetitions": stability.get("fitted_repetitions"),
                "repetitions": stability.get("repetitions"),
            },
            "local_photometric": rescue.get("local_photometric"),
            "local_map_shape": rescue.get("local_map_shape"),
        },
    }


def compact_predicate(score: dict[str, Any]) -> dict[str, Any]:
    return {
        "observable": score.get("observable"),
        "accepted": bool(score.get("accepted", False)),
        "reason": score.get("reason"),
        "core_pixels": score.get("core_pixels"),
        "source_separation": score.get("source_separation"),
        "pre_error": score.get("pre_error"),
        "post_error": score.get("post_error"),
        "normalized_margin": score.get("normalized_margin"),
        "context_error": score.get("context_error"),
    }


def confidently_observed_post_state(predicate: dict[str, Any] | None) -> bool:
    """Return whether an aligned query confidently exhibits the stored effect.

    Progress recovery uses the negative mirror of the frozen P_pre margin and
    the same stable-context threshold.  It never authorizes an action; it only
    lets the selector continue to a successor, which must independently pass
    S, G, and P_pre.
    """
    if not predicate or not predicate.get("observable"):
        return False
    margin = predicate.get("normalized_margin")
    context_error = predicate.get("context_error")
    if margin is None or context_error is None:
        return False
    return bool(
        not predicate.get("accepted", False)
        and float(margin) <= -PRECONDITION.MIN_NORMALIZED_MARGIN
        and (float(context_error) <= PRECONDITION.MAX_CONTEXT_ERROR)
    )


def compact_frontend(receipt: dict[str, Any]) -> dict[str, Any]:
    scores = receipt.get("match_scores")
    return {
        "frontend_version": receipt.get("frontend_version"),
        "source_image_array_sha256": receipt.get("source_image_array_sha256"),
        "current_image_array_sha256": receipt.get("current_image_array_sha256"),
        "source_keypoints": receipt.get("source_keypoints"),
        "target_keypoints": receipt.get("target_keypoints"),
        "matches": receipt.get("matches"),
        "match_score_min": float(np.min(scores)) if scores is not None and len(scores) else None,
        "match_score_median": float(np.median(scores)) if scores is not None and len(scores) else None,
        "match_score_max": float(np.max(scores)) if scores is not None and len(scores) else None,
        "inference_s": receipt.get("inference_s"),
        "cache_hit": receipt.get("cache_hit", False),
        "cache_lookup_s": receipt.get("cache_lookup_s"),
    }


def source_effect_receipt(transition: Transition) -> dict[str, Any]:
    return compact_predicate(
        {
            **PRECONDITION.effect_model(transition.before, transition.after),
            "accepted": False,
            "reason": "source_effect_observability_only",
            "pre_error": None,
            "post_error": None,
            "normalized_margin": None,
            "context_error": None,
        }
    )


def apply_learned_consensus_rescue(receipt: dict[str, Any]) -> dict[str, Any]:
    """Relax only the local angular-bin check for pinned learned matches.

    Actions near a screen edge cannot surround themselves with matches in many
    angular sectors even when the learned correspondences establish a stable,
    low-residual global map.  Every other existing geometric, photometric,
    stability, local-support, shape, and boundary check remains mandatory.
    """
    rescue = receipt.get("action_conditioned_rescue") or {}
    full_fit = rescue.get("full_fit") or {}
    geometric = rescue.get("geometric_branch") or {}
    geometric_checks = geometric.get("checks") or {}
    stability = rescue.get("stability") or {}
    shape = rescue.get("local_map_shape") or {}
    matrix = rescue.get("candidate_matrix")
    full_fit_checks = full_fit.get("checks") or {}
    checks = {
        "eligible_action_conditioned_candidate": bool(rescue.get("eligible")),
        "all_full_fit_checks": bool(full_fit_checks)
        and all((bool(value) for value in full_fit_checks.values())),
        "minimum_local_inliers": bool(geometric_checks.get("minimum_local_inliers")),
        "maximum_local_p90_residual": bool(geometric_checks.get("maximum_local_p90_residual")),
        "maximum_nearest_inlier_fraction": bool(geometric_checks.get("maximum_nearest_inlier_fraction")),
        "loose_local_photometric": bool(geometric_checks.get("loose_local_photometric")),
        "action_stability": bool(stability.get("pass")),
        "local_map_shape": bool(shape.get("pass")),
        "mapped_action_in_target_image_bounds": bool(rescue.get("mapped_action_in_target_image_bounds")),
        "finite_candidate_matrix": bool(
            matrix is not None
            and np.asarray(matrix).shape == (3, 3)
            and np.isfinite(np.asarray(matrix, dtype=np.float64)).all()
        ),
    }
    accepted = all(checks.values())
    audit = {
        "eligible": True,
        "accepted": accepted,
        "reason": "all_checks_except_local_angular_bins_pass"
        if accepted
        else "learned_consensus_checks_failed",
        "omitted_parent_check": "minimum_angular_bins",
        "checks": checks,
        "candidate_matrix": matrix,
        "mapped_action": rescue.get("mapped_action"),
    }
    receipt["learned_consensus_rescue"] = audit
    if accepted:
        receipt["supported"] = True
        receipt["matrix"] = matrix
        receipt["reason"] = "learned_consensus_action_certificate"
        receipt["decision_path"] = "learned_consensus_rescue"
    return receipt


def apply_matrix(matrix: list[list[float]], point: tuple[float, float]) -> tuple[float, float]:
    transform = np.asarray(matrix, dtype=np.float64)
    homogeneous = transform @ np.asarray([point[0], point[1], 1.0])
    if abs(float(homogeneous[2])) <= 1e-12:
        return (float("nan"), float("nan"))
    return (float(homogeneous[0] / homogeneous[2]), float(homogeneous[1] / homogeneous[2]))


def warp(image: np.ndarray, matrix: list[list[float]], target: np.ndarray) -> np.ndarray:
    return cv2.warpPerspective(
        image,
        np.asarray(matrix, dtype=np.float64),
        (target.shape[1], target.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(255, 255, 255),
    )


def point_inside(point: tuple[float, float], image: np.ndarray) -> bool:
    return bool(
        all((math.isfinite(value) for value in point))
        and 0.0 <= point[0] < image.shape[1]
        and (0.0 <= point[1] < image.shape[0])
    )


def reject_receipt(
    transition: Transition,
    gate: str,
    reason: str,
    geometry: dict[str, Any] | None = None,
    precondition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "method_version": METHOD_VERSION,
        "transition_id": transition.transition_id,
        "demonstration_order": transition.demonstration_order,
        "authorized": False,
        "decision": "fallback",
        "failed_gate": gate,
        "reason": reason,
        "source_success": transition.source_success,
        "geometry": geometry,
        "precondition": precondition,
        "mapped_action_xy": None,
    }


def authorize(
    transition: Transition,
    current: np.ndarray,
    *,
    require_precondition: bool = True,
    match_provider: MatchProvider | None = None,
) -> tuple[dict[str, Any], AuthorizationContext | None]:
    """Evaluate S, G, and optionally P_pre for one transition."""
    if current.ndim != 3 or current.shape[2] != 3:
        raise ValueError("current must be an HxWx3 image")
    if not transition.source_success:
        return (reject_receipt(transition, "S", "source_episode_not_successful"), None)
    frontend_receipt = None
    if match_provider is None:
        raw_geometry = GEOMETRY.register_action_contract(
            transition.before, current, list(transition.action_xy)
        )
    else:
        frontend_receipt = match_provider(transition.before, current)
        raw_geometry = GEOMETRY.register_action_contract(
            transition.before, current, list(transition.action_xy), matcher=learned_matcher(frontend_receipt)
        )
        if not raw_geometry.get("supported"):
            raw_geometry = apply_learned_consensus_rescue(raw_geometry)
        raw_geometry["correspondence_frontend"] = compact_frontend(frontend_receipt)
    geometry = compact_geometry(raw_geometry)
    if not geometry["supported"]:
        return (
            reject_receipt(
                transition, "G", str(geometry.get("reason") or "geometry_unsupported"), geometry=geometry
            ),
            None,
        )
    matrix = geometry["matrix"]
    if matrix is None:
        raise RuntimeError("supported geometry receipt has no matrix")
    mapped_action = apply_matrix(matrix, transition.action_xy)
    if not point_inside(mapped_action, current):
        return (
            reject_receipt(transition, "G", "mapped_action_outside_current_image", geometry=geometry),
            None,
        )
    warped_before = warp(transition.before, matrix, current)
    warped_after = warp(transition.after, matrix, current)
    if require_precondition:
        precondition = _precondition(transition, warped_before, warped_after, current)
        if not precondition["accepted"]:
            return (
                reject_receipt(
                    transition,
                    "P_pre",
                    str(precondition.get("reason") or "precondition_rejected"),
                    geometry=geometry,
                    precondition=precondition,
                ),
                None,
            )
    else:
        precondition = {"observable": None, "accepted": None, "reason": "disabled_by_ablation"}
    receipt = {
        "method_version": METHOD_VERSION,
        "transition_id": transition.transition_id,
        "demonstration_order": transition.demonstration_order,
        "authorized": True,
        "decision": "reuse",
        "failed_gate": None,
        "reason": "S_and_G_and_P_pre" if require_precondition else "S_and_G",
        "source_success": True,
        "geometry": geometry,
        "precondition": precondition,
        "mapped_action_xy": list(mapped_action),
    }
    context = AuthorizationContext(
        transition_id=transition.transition_id,
        demonstration_order=transition.demonstration_order,
        mapped_action_xy=mapped_action,
        warped_before=warped_before,
        warped_after=warped_after,
        authorization_receipt=receipt,
        terminal=transition.terminal,
    )
    return (receipt, context)


def verify_post(context: AuthorizationContext, observed_next: np.ndarray) -> dict[str, Any]:
    """Verify P_post by exchanging the stored pre/post roles."""
    if observed_next.shape != context.warped_after.shape:
        raise ValueError("postcondition image shape differs from the authorization image")
    raw = PRECONDITION.score_query(context.warped_after, context.warped_before, observed_next)
    if (
        context.terminal
        and (not raw["accepted"])
        and (raw.get("reason") == "context_mismatch")
        and (raw.get("normalized_margin") is not None)
        and (float(raw["normalized_margin"]) >= PRECONDITION.MIN_NORMALIZED_MARGIN)
    ):
        raw = {**raw, "accepted": True, "reason": "terminal_post_margin_with_confirmed_prefix"}
    predicate = compact_predicate(raw)
    return {
        "method_version": METHOD_VERSION,
        "transition_id": context.transition_id,
        "demonstration_order": context.demonstration_order,
        "post_verified": predicate["accepted"],
        "decision": "continue_replay" if predicate["accepted"] else "force_fallback",
        "postcondition": predicate,
    }


def learned_matcher(receipt: dict[str, Any]):
    """Validate one learned correspondence result and return an isolated provider."""
    source_points = np.asarray(receipt["source_points"], dtype=np.float64)
    target_points = np.asarray(receipt["target_points"], dtype=np.float64)
    if (
        source_points.shape != target_points.shape
        or source_points.ndim != 2
        or source_points.shape[1:] != (2,)
        or (not np.isfinite(source_points).all())
        or (not np.isfinite(target_points).all())
    ):
        raise ValueError("correspondence provider returned invalid point arrays")
    counts = {
        "source_keypoints": int(receipt["source_keypoints"]),
        "target_keypoints": int(receipt["target_keypoints"]),
        "mutual_ratio_matches": int(receipt["matches"]),
    }
    if counts["mutual_ratio_matches"] != len(source_points):
        raise ValueError("correspondence count does not match point arrays")

    def match(_source, _target):
        return (source_points.copy(), target_points.copy(), dict(counts))

    return match


def _precondition(transition, warped_before, warped_after, current):
    raw_precondition = PRECONDITION.score_query(warped_before, warped_after, current)
    if (
        transition.terminal
        and (not raw_precondition["accepted"])
        and (raw_precondition.get("reason") == "context_mismatch")
        and (raw_precondition.get("normalized_margin") is not None)
        and (float(raw_precondition["normalized_margin"]) >= PRECONDITION.MIN_NORMALIZED_MARGIN)
    ):
        raw_precondition = {
            **raw_precondition,
            "accepted": True,
            "reason": "terminal_pre_margin_with_confirmed_prefix",
        }
    return compact_predicate(raw_precondition)
