"""Read-only checks performed before loading services or creating run directories."""

from __future__ import annotations

import importlib.util
import json
import math
import os
from typing import Any

import cv2

from dafc.configuration import RunConfig, input_problems
from dafc.resources import sha256_file

REQUIRED = {
    "routing": {"source", "target"},
    "browser-episode": {"chromedriver", "checkpoint_attestation"},
    "android-episode": {"adb", "a11y_apk", "checkpoint_attestation"},
    "android-atomic": {"adb", "a11y_apk", "source_registry", "checkpoint_attestation"},
}
PARAMETERS = {
    "routing": {"policy", "source_success", "action_xy"},
    "browser-episode": {
        "resource_base",
        "task",
        "seed",
        "contract",
        "policy",
        "actor",
        "memory_enable_after_steps",
        "require_risk_certificate",
        "score_postcondition_for_ablations",
    },
    "android-episode": {
        "resource_base",
        "task",
        "contract",
        "policy",
        "actor",
        "timer_params",
        "browser_task_seed",
        "serial",
        "console_port",
        "grpc_port",
        "require_risk_certificate",
        "score_postcondition_for_ablations",
    },
    "android-atomic": {"actor", "serial", "console_port", "grpc_port", "limit", "resource_base"},
}
PINNED = {
    "a11y_apk": "97a56a544e44d79f9b3181fc7dbdd72cffa908efd3d53c82afad1773061a350a",
    "chromedriver": "51df048171813deb307f7bd26baf49db030032f9332cedce8ce2797e03b86be1",
}


def _package_problems(config: RunConfig) -> list[str]:
    if config.workflow == "routing":
        return []
    packages = (
        {"miniwob": "uv sync --extra browser"}
        if config.workflow == "browser-episode"
        else {
            "android_world": "restore AndroidWorld at the catalogued revision and install its declared dependencies into this .venv",
            "android_env": "install the AndroidWorld dependency group from its pinned source",
        }
    )
    return [
        f"Optional dependency {name} is missing; {step}"
        for name, step in packages.items()
        if importlib.util.find_spec(name) is None
    ]


def _runtime_inputs(config: RunConfig) -> list[str]:
    problems = []
    for name, digest in PINNED.items():
        item = config.inputs.get(name)
        if item and item.path.is_file() and sha256_file(item.path) != digest:
            problems.append(f"Pinned {name} checksum mismatch: {item.path}; restore the catalogue version")
    for name in ("adb", "chromedriver"):
        item = config.inputs.get(name)
        if item and item.path.is_file() and not os.access(item.path, os.X_OK):
            problems.append(f"Runtime executable lacks execute permission: {item.path}")
    if config.workflow != "routing" and "model" not in config.services:
        problems.append("services.model is required; start the documented actor service explicitly")
    if config.parameters.get("policy") in {"full", "sg", "sgp"} and "lightglue" not in config.services:
        problems.append("services.lightglue is required for configured guarded replay")
    return problems


def validate_parameters(config: RunConfig) -> None:
    p = config.parameters
    from dafc.jobs.parameter_types import validate_types

    validate_types(config)
    unknown = set(p) - PARAMETERS[config.workflow]
    if unknown:
        raise ValueError(f"Unknown {config.workflow} parameters: {sorted(unknown)}")
    if config.workflow == "routing":
        if p.get("policy", "dafc") not in {"always", "s_only", "standard_ransac", "dafc"}:
            raise ValueError("routing policy must be always, s_only, standard_ransac or dafc")
        if not isinstance(p.get("source_success"), bool):
            raise ValueError("source_success must be boolean")
        action = p.get("action_xy")
        if not isinstance(action, list) or len(action) != 2:
            raise ValueError("action_xy requires two pixel coordinates")
        if any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(float(x))
            for x in action
        ):
            raise ValueError("action_xy must contain two finite numbers")
    else:
        if p.get("actor") not in {"qwen25vl", "uitars"}:
            raise ValueError("actor must be qwen25vl or uitars")
        if config.workflow != "android-atomic" and p.get("policy") not in {
            "always",
            "raw",
            "naive",
            "sg",
            "sgp",
            "full",
        }:
            raise ValueError("episode policy must be always, raw, naive, sg, sgp or full")


def prepare(config: RunConfig, *, check_packages: bool = True) -> dict[str, Any]:
    validate_parameters(config)
    problems = input_problems(config)
    problems += [
        f"inputs.{name} is required for {config.workflow}"
        for name in sorted(REQUIRED[config.workflow] - config.inputs.keys())
    ]
    if check_packages:
        problems += _package_problems(config)
    problems += _runtime_inputs(config)
    if problems:
        return {"problems": problems}
    if config.workflow == "routing":
        for name in ("source", "target"):
            if cv2.imread(str(config.inputs[name].path), cv2.IMREAD_COLOR) is None:
                problems.append(f"Input {name} is not a decodable image: {config.inputs[name].path}")
        return {"problems": problems}
    expected = json.loads(config.inputs["checkpoint_attestation"].path.read_text())
    if not isinstance(expected, dict) or not expected:
        raise ValueError(
            "checkpoint_attestation must be the nonempty checkpoint object from a trusted actor health receipt"
        )
    from dafc.jobs.atomic import prepare_atomic
    from dafc.jobs.platforms import prepare_episode

    extra = prepare_atomic(config) if config.workflow == "android-atomic" else prepare_episode(config)
    return {"problems": problems, "expected_checkpoint": expected, **extra}
