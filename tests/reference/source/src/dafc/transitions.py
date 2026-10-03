#!/usr/bin/env python3
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
from importlib import import_module
from dafc.optional import LazyModule

import importlib.util
import math
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

import cv2
import numpy as np


from dafc.paths import PROJECT
SCRIPTS = PROJECT / "scripts"
METHOD_VERSION = "transition-guard-v2.4-progress-recovery"
MATCH_FRONTEND_LOCK = threading.Lock()
MatchProvider = Callable[[np.ndarray, np.ndarray], dict[str, Any]]


def binomial_cdf(successes: int, trials: int, probability: float) -> float:
    """Return P[X <= successes] for X ~ Binomial(trials, probability)."""

    if trials < 0 or not 0 <= successes <= trials:
        raise ValueError("binomial counts are invalid")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("binomial probability must be in [0, 1]")
    if successes == trials or probability == 0.0:
        return 1.0
    if probability == 1.0:
        return 0.0
    log_p = math.log(probability)
    log_q = math.log1p(-probability)
    terms = [
        math.lgamma(trials + 1)
        - math.lgamma(index + 1)
        - math.lgamma(trials - index + 1)
        + index * log_p
        + (trials - index) * log_q
        for index in range(successes + 1)
    ]
    anchor = max(terms)
    return min(1.0, math.exp(anchor) * sum(math.exp(row - anchor) for row in terms))


def clopper_pearson_upper(
    harmful: int, authorized: int, confidence: float
) -> float:
    """One-sided exact binomial upper confidence bound."""

    if authorized <= 0 or not 0 <= harmful <= authorized:
        raise ValueError("risk counts require 0 <= harmful <= authorized")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    if harmful == authorized:
        return 1.0
    alpha = 1.0 - confidence
    if harmful == 0:
        return 1.0 - alpha ** (1.0 / authorized)
    lower = harmful / authorized
    upper = 1.0
    for _ in range(100):
        midpoint = (lower + upper) / 2.0
        if binomial_cdf(harmful, authorized, midpoint) > alpha:
            lower = midpoint
        else:
            upper = midpoint
    return upper


@dataclass(frozen=True)
class RiskCertificate:
    """Held-out activation certificate for the frozen admission rule.

    This certificate does not make per-action safety claims.  It only enables
    runtime replay when a separately labelled calibration audit is large
    enough and its exact one-sided harmful-authorization bound meets the
    declared deployment budget.  Its statistical interpretation requires the
    declared calibration unit to be exchangeable with deployment queries.
    """

    certificate_id: str
    method_version: str
    calibration_unit: str
    calibration_sha256: str
    confidence: float
    risk_limit: float
    minimum_authorizations: int
    authorizations: int
    harmful_authorizations: int
    upper_bound: float

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "RiskCertificate":
        certificate = cls(
            certificate_id=str(row["certificate_id"]),
            method_version=str(row["method_version"]),
            calibration_unit=str(row["calibration_unit"]),
            calibration_sha256=str(row["calibration_sha256"]),
            confidence=float(row["confidence"]),
            risk_limit=float(row["risk_limit"]),
            minimum_authorizations=int(row["minimum_authorizations"]),
            authorizations=int(row["authorizations"]),
            harmful_authorizations=int(row["harmful_authorizations"]),
            upper_bound=float(row["upper_bound"]),
        )
        certificate.validate()
        return certificate

    def validate(self) -> None:
        if not self.certificate_id or not self.calibration_unit:
            raise ValueError("risk certificate identity fields are required")
        if len(self.calibration_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.calibration_sha256
        ):
            raise ValueError("calibration_sha256 is not lowercase SHA-256")
        if not 0.0 < self.risk_limit < 1.0:
            raise ValueError("risk_limit must lie strictly between zero and one")
        if self.minimum_authorizations <= 0:
            raise ValueError("minimum_authorizations must be positive")
        recomputed = clopper_pearson_upper(
            self.harmful_authorizations,
            self.authorizations,
            self.confidence,
        )
        if not math.isclose(self.upper_bound, recomputed, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("risk certificate upper bound is not reproducible")

    @property
    def active(self) -> bool:
        return bool(
            self.method_version == METHOD_VERSION
            and self.authorizations >= self.minimum_authorizations
            and self.upper_bound < self.risk_limit
        )

    def receipt(self) -> dict[str, Any]:
        return {
            "certificate_id": self.certificate_id,
            "method_version": self.method_version,
            "calibration_unit": self.calibration_unit,
            "calibration_sha256": self.calibration_sha256,
            "confidence": self.confidence,
            "risk_limit": self.risk_limit,
            "minimum_authorizations": self.minimum_authorizations,
            "authorizations": self.authorizations,
            "harmful_authorizations": self.harmful_authorizations,
            "upper_bound": self.upper_bound,
            "active": self.active,
        }


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
GEOMETRY = import_module('dafc.registration.action_contract')
PRECONDITION = import_module('dafc.adapters.browser.c431_transition_precondition_development')


@dataclass(frozen=True)
class Transition:
    """One source transition admitted from a completed successful episode."""

    transition_id: str
    demonstration_order: int
    source_success: bool
    before: np.ndarray
    action_xy: tuple[float, float]
    after: np.ndarray
    terminal: bool = False

    def __post_init__(self) -> None:
        if not self.transition_id:
            raise ValueError("transition_id must be non-empty")
        if self.demonstration_order < 0:
            raise ValueError("demonstration_order must be non-negative")
        if self.before.ndim != 3 or self.before.shape[2] != 3:
            raise ValueError("before must be an HxWx3 image")
        if self.after.shape != self.before.shape:
            raise ValueError("before and after images must have the same shape")
        if not all(math.isfinite(value) for value in self.action_xy):
            raise ValueError("action_xy must be finite")


@dataclass
class AuthorizationContext:
    """Non-serializable templates retained only until post-action checking."""

    transition_id: str
    demonstration_order: int
    mapped_action_xy: tuple[float, float]
    warped_before: np.ndarray
    warped_after: np.ndarray
    authorization_receipt: dict[str, Any]
    terminal: bool = False


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
            "nearest_inlier_distance_fraction": rescue.get(
                "nearest_inlier_distance_fraction"
            ),
            "geometric_branch": rescue.get("geometric_branch"),
            "appearance_branch": rescue.get("appearance_branch"),
            "stability": {
                "pass": stability.get("pass"),
                "maximum_disagreement_px": stability.get(
                    "maximum_disagreement_px"
                ),
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


def confidently_observed_post_state(
    predicate: dict[str, Any] | None,
) -> bool:
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
        and float(context_error) <= PRECONDITION.MAX_CONTEXT_ERROR
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
        "match_score_min": (
            float(np.min(scores)) if scores is not None and len(scores) else None
        ),
        "match_score_median": (
            float(np.median(scores)) if scores is not None and len(scores) else None
        ),
        "match_score_max": (
            float(np.max(scores)) if scores is not None and len(scores) else None
        ),
        "inference_s": receipt.get("inference_s"),
        "cache_hit": receipt.get("cache_hit", False),
        "cache_lookup_s": receipt.get("cache_lookup_s"),
    }


def match_modules() -> list[Any]:
    """Locate every loaded copy of the frozen classical match boundary."""

    target_path = (SCRIPTS / "c357_visual_contract_registration_pilot.py").resolve()
    modules = []
    for module in list(sys.modules.values()):
        module_file = getattr(module, "__file__", None)
        if module_file is None or not hasattr(module, "mutual_ratio_matches"):
            continue
        try:
            path = Path(module_file).resolve()
        except (OSError, TypeError):
            continue
        if path == target_path:
            modules.append(module)
    if len(modules) < 2:
        raise RuntimeError(
            f"expected all frozen registration match modules, found {len(modules)}"
        )
    return modules


@contextmanager
def injected_correspondences(receipt: dict[str, Any]) -> Iterator[None]:
    """Temporarily inject one learned match set into the frozen G certificate."""

    source_points = np.asarray(receipt["source_points"], dtype=np.float64)
    target_points = np.asarray(receipt["target_points"], dtype=np.float64)
    if (
        source_points.shape != target_points.shape
        or source_points.ndim != 2
        or source_points.shape[1:] != (2,)
        or not np.isfinite(source_points).all()
        or not np.isfinite(target_points).all()
    ):
        raise ValueError("correspondence provider returned invalid point arrays")
    counts = {
        "source_keypoints": int(receipt["source_keypoints"]),
        "target_keypoints": int(receipt["target_keypoints"]),
        "mutual_ratio_matches": int(receipt["matches"]),
    }

    def learned_matches(_source: np.ndarray, _target: np.ndarray):
        return source_points.copy(), target_points.copy(), dict(counts)

    with MATCH_FRONTEND_LOCK:
        modules = match_modules()
        originals = [(module, module.mutual_ratio_matches) for module in modules]
        try:
            for module, _ in originals:
                module.mutual_ratio_matches = learned_matches
            yield
        finally:
            for module, original in originals:
                module.mutual_ratio_matches = original


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
        and all(bool(value) for value in full_fit_checks.values()),
        "minimum_local_inliers": bool(
            geometric_checks.get("minimum_local_inliers")
        ),
        "maximum_local_p90_residual": bool(
            geometric_checks.get("maximum_local_p90_residual")
        ),
        "maximum_nearest_inlier_fraction": bool(
            geometric_checks.get("maximum_nearest_inlier_fraction")
        ),
        "loose_local_photometric": bool(
            geometric_checks.get("loose_local_photometric")
        ),
        "action_stability": bool(stability.get("pass")),
        "local_map_shape": bool(shape.get("pass")),
        "mapped_action_in_target_image_bounds": bool(
            rescue.get("mapped_action_in_target_image_bounds")
        ),
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
        "reason": (
            "all_checks_except_local_angular_bins_pass"
            if accepted
            else "learned_consensus_checks_failed"
        ),
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
    return (
        float(homogeneous[0] / homogeneous[2]),
        float(homogeneous[1] / homogeneous[2]),
    )


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
        all(math.isfinite(value) for value in point)
        and 0.0 <= point[0] < image.shape[1]
        and 0.0 <= point[1] < image.shape[0]
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
        return reject_receipt(
            transition, "S", "source_episode_not_successful"
        ), None

    frontend_receipt = None
    if match_provider is None:
        raw_geometry = GEOMETRY.register_action_contract(
            transition.before,
            current,
            list(transition.action_xy),
        )
    else:
        frontend_receipt = match_provider(transition.before, current)
        with injected_correspondences(frontend_receipt):
            raw_geometry = GEOMETRY.register_action_contract(
                transition.before,
                current,
                list(transition.action_xy),
            )
        if not raw_geometry.get("supported"):
            raw_geometry = apply_learned_consensus_rescue(raw_geometry)
        raw_geometry["correspondence_frontend"] = compact_frontend(
            frontend_receipt
        )
    geometry = compact_geometry(raw_geometry)
    if not geometry["supported"]:
        return reject_receipt(
            transition,
            "G",
            str(geometry.get("reason") or "geometry_unsupported"),
            geometry=geometry,
        ), None
    matrix = geometry["matrix"]
    if matrix is None:
        raise RuntimeError("supported geometry receipt has no matrix")
    mapped_action = apply_matrix(matrix, transition.action_xy)
    if not point_inside(mapped_action, current):
        return reject_receipt(
            transition,
            "G",
            "mapped_action_outside_current_image",
            geometry=geometry,
        ), None

    warped_before = warp(transition.before, matrix, current)
    warped_after = warp(transition.after, matrix, current)
    if require_precondition:
        raw_precondition = PRECONDITION.score_query(
            warped_before, warped_after, current
        )
        if (
            transition.terminal
            and not raw_precondition["accepted"]
            and raw_precondition.get("reason") == "context_mismatch"
            and raw_precondition.get("normalized_margin") is not None
            and float(raw_precondition["normalized_margin"])
            >= PRECONDITION.MIN_NORMALIZED_MARGIN
        ):
            raw_precondition = {
                **raw_precondition,
                "accepted": True,
                "reason": "terminal_pre_margin_with_confirmed_prefix",
            }
        precondition = compact_predicate(raw_precondition)
        if not precondition["accepted"]:
            return reject_receipt(
                transition,
                "P_pre",
                str(precondition.get("reason") or "precondition_rejected"),
                geometry=geometry,
                precondition=precondition,
            ), None
    else:
        precondition = {
            "observable": None,
            "accepted": None,
            "reason": "disabled_by_ablation",
        }

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
    return receipt, context


def select(
    transitions: Iterable[Transition],
    current: np.ndarray,
    *,
    require_precondition: bool = True,
    confirmed_transition_ids: Iterable[str] = (),
    match_provider: MatchProvider | None = None,
    risk_certificate: RiskCertificate | None = None,
    require_risk_certificate: bool = False,
) -> tuple[dict[str, Any], AuthorizationContext | None]:
    """Select the earliest applicable, not-yet-confirmed transition.

    A source effect that is not visually observable blocks that transition and
    every dependent successor in the demonstration.  This prevents a later
    action from being replayed across a hidden source-state change.  The
    S+G-only ablation disables this prefix rule together with P_pre.
    """

    ordered = sorted(transitions, key=lambda row: row.demonstration_order)
    known_ids = {row.transition_id for row in ordered}
    if len(known_ids) != len(ordered):
        raise ValueError("transition IDs must be unique")
    confirmed = set(confirmed_transition_ids)
    progress_recovered: set[str] = set()
    unknown_confirmed = confirmed - known_ids
    if unknown_confirmed:
        raise ValueError(
            f"confirmed transition IDs are absent from memory: {unknown_confirmed}"
        )
    risk_control = (
        risk_certificate.receipt() if risk_certificate is not None else None
    )
    if require_risk_certificate and (
        risk_certificate is None or not risk_certificate.active
    ):
        return {
            "method_version": METHOD_VERSION,
            "decision": "fallback",
            "selected_transition_id": None,
            "selected_demonstration_order": None,
            "mapped_action_xy": None,
            "confirmed_transition_ids": sorted(confirmed),
            "progress_recovered_transition_ids": [],
            "observable_prefix_intact": None,
            "evaluations": [],
            "risk_control": risk_control,
            "fallback_reason": (
                "risk_certificate_missing"
                if risk_certificate is None
                else "risk_certificate_inactive"
            ),
        }, None
    evaluations = []
    observable_prefix = True
    replayable_predecessors: set[str] = set()
    for transition in ordered:
        source_effect = source_effect_receipt(transition)
        effect_observable = bool(source_effect["observable"])
        terminal_prefix_missing = bool(
            require_precondition
            and transition.terminal
            and not replayable_predecessors.issubset(confirmed)
        )
        if require_precondition and not observable_prefix:
            receipt = reject_receipt(
                transition,
                "P_pre",
                "unobservable_source_predecessor",
                precondition={
                    **source_effect,
                    "accepted": False,
                    "reason": "unobservable_source_predecessor",
                },
            )
            context = None
        elif terminal_prefix_missing:
            receipt = reject_receipt(
                transition,
                "H",
                "terminal_transition_requires_confirmed_prefix",
                precondition=(source_effect if require_precondition else None),
            )
            context = None
        elif require_precondition and not effect_observable:
            receipt = reject_receipt(
                transition,
                "P_pre",
                "unobservable_source_effect",
                precondition={
                    **source_effect,
                    "accepted": False,
                    "reason": "unobservable_source_effect",
                },
            )
            context = None
        elif transition.transition_id in confirmed:
            receipt = reject_receipt(
                transition,
                "H",
                "transition_already_post_verified",
                precondition=(source_effect if require_precondition else None),
            )
            context = None
        else:
            receipt, context = authorize(
                transition,
                current,
                require_precondition=require_precondition,
                match_provider=match_provider,
            )
            if (
                require_precondition
                and not transition.terminal
                and context is None
                and confidently_observed_post_state(receipt.get("precondition"))
            ):
                receipt = {
                    **receipt,
                    "decision": "advance_progress",
                    "failed_gate": None,
                    "reason": "confident_post_state_observed",
                    "progress_recovered": True,
                }
                confirmed.add(transition.transition_id)
                progress_recovered.add(transition.transition_id)
        evaluations.append(receipt)
        if require_precondition and not effect_observable:
            observable_prefix = False
        if effect_observable and not transition.terminal:
            replayable_predecessors.add(transition.transition_id)
        if context is not None:
            return {
                "method_version": METHOD_VERSION,
                "decision": "reuse",
                "selected_transition_id": transition.transition_id,
                "selected_demonstration_order": transition.demonstration_order,
                "mapped_action_xy": list(context.mapped_action_xy),
                "confirmed_transition_ids": sorted(confirmed),
                "progress_recovered_transition_ids": sorted(progress_recovered),
                "observable_prefix_intact": observable_prefix,
                "evaluations": evaluations,
                "risk_control": risk_control,
            }, context
    return {
        "method_version": METHOD_VERSION,
        "decision": "fallback",
        "selected_transition_id": None,
        "selected_demonstration_order": None,
        "mapped_action_xy": None,
        "confirmed_transition_ids": sorted(confirmed),
        "progress_recovered_transition_ids": sorted(progress_recovered),
        "observable_prefix_intact": observable_prefix,
        "evaluations": evaluations,
        "risk_control": risk_control,
    }, None


def verify_post(
    context: AuthorizationContext, observed_next: np.ndarray
) -> dict[str, Any]:
    """Verify P_post by exchanging the stored pre/post roles."""

    if observed_next.shape != context.warped_after.shape:
        raise ValueError(
            "postcondition image shape differs from the authorization image"
        )
    raw = PRECONDITION.score_query(
        context.warped_after,
        context.warped_before,
        observed_next,
    )
    if (
        context.terminal
        and not raw["accepted"]
        and raw.get("reason") == "context_mismatch"
        and raw.get("normalized_margin") is not None
        and float(raw["normalized_margin"])
        >= PRECONDITION.MIN_NORMALIZED_MARGIN
    ):
        raw = {
            **raw,
            "accepted": True,
            "reason": "terminal_post_margin_with_confirmed_prefix",
        }
    predicate = compact_predicate(raw)
    return {
        "method_version": METHOD_VERSION,
        "transition_id": context.transition_id,
        "demonstration_order": context.demonstration_order,
        "post_verified": predicate["accepted"],
        "decision": "continue_replay" if predicate["accepted"] else "force_fallback",
        "postcondition": predicate,
    }
