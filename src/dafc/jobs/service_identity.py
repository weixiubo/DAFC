"""Validate a learned matcher receipt against the declared scientific implementation."""

from typing import Any

from dafc.models.lightglue_runtime_frontend import (
    EXPECTED_REPO_COMMIT,
    EXPECTED_WEIGHT_HASHES,
    FRONTEND_VERSION,
)
from dafc.models.lightglue_runtime_service import SERVICE_VERSION


def validate_lightglue(health: dict[str, Any]) -> dict[str, Any]:
    bindings = health.get("bindings", {})
    if (
        health.get("service_version") != SERVICE_VERSION
        or bindings.get("frontend_version") != FRONTEND_VERSION
    ):
        raise ValueError("LightGlue service implementation does not match the declared version")
    if bindings.get("repo", {}).get("commit") != EXPECTED_REPO_COMMIT:
        raise ValueError("LightGlue source revision mismatch")
    weights = bindings.get("weights", {})
    for name, expected in EXPECTED_WEIGHT_HASHES.items():
        if weights.get(name, {}).get("sha256") != expected:
            raise ValueError(f"LightGlue weight checksum mismatch: {name}")
    return {"service_version": SERVICE_VERSION, "bindings": bindings}
