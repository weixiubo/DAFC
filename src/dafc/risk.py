from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from dafc.transition_types import METHOD_VERSION


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
    return min(1.0, math.exp(anchor) * sum((math.exp(row - anchor) for row in terms)))


def clopper_pearson_upper(harmful: int, authorized: int, confidence: float) -> float:
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
            (char not in "0123456789abcdef" for char in self.calibration_sha256)
        ):
            raise ValueError("calibration_sha256 is not lowercase SHA-256")
        if not 0.0 < self.risk_limit < 1.0:
            raise ValueError("risk_limit must lie strictly between zero and one")
        if self.minimum_authorizations <= 0:
            raise ValueError("minimum_authorizations must be positive")
        recomputed = clopper_pearson_upper(self.harmful_authorizations, self.authorizations, self.confidence)
        if not math.isclose(self.upper_bound, recomputed, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("risk certificate upper bound is not reproducible")

    @property
    def active(self) -> bool:
        return bool(
            self.method_version == METHOD_VERSION
            and self.authorizations >= self.minimum_authorizations
            and (self.upper_bound < self.risk_limit)
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
