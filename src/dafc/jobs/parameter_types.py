"""Validate public configuration values before any experiment is started."""

from typing import Any

from dafc.configuration import RunConfig


def validate_types(config: RunConfig) -> None:
    p = config.parameters
    for key in ("task", "actor", "policy", "contract", "serial", "resource_base"):
        if key in p and (not isinstance(p[key], str) or not p[key]):
            raise ValueError(f"parameters.{key} must be a nonempty string")
    for key in ("source_success", "require_risk_certificate", "score_postcondition_for_ablations"):
        if key in p and not isinstance(p[key], bool):
            raise ValueError(f"parameters.{key} must be boolean")
    for key in (
        "seed",
        "browser_task_seed",
        "memory_enable_after_steps",
        "console_port",
        "grpc_port",
        "limit",
    ):
        if key in p and (isinstance(p[key], bool) or not isinstance(p[key], int)):
            raise ValueError(f"parameters.{key} must be an integer")
    if "timer_params" in p and not isinstance(p["timer_params"], dict):
        raise ValueError("timer_params must be an hours/minutes/seconds mapping")
    _validate_replay(p)
    _validate_device(p)


def _validate_replay(p: dict[str, Any]) -> None:
    if p.get("require_risk_certificate") and p.get("policy") != "full":
        raise ValueError("risk-limited activation applies only to full policy")
    if p.get("score_postcondition_for_ablations") and p.get("policy") not in {"full", "sg", "sgp"}:
        raise ValueError("scoring-only postconditions require a guarded replay policy")
    prefix = p.get("memory_enable_after_steps", 0)
    if prefix < 0 or (p.get("policy") == "always" and prefix != 0):
        raise ValueError("memory warm-start must be nonnegative and zero for always policy")


def _validate_device(p: dict[str, Any]) -> None:
    for key in ("console_port", "grpc_port"):
        if key in p and not 1 <= p[key] <= 65535:
            raise ValueError(f"{key} must be a port from 1 to 65535")
    if "serial" in p and p["serial"] != f"emulator-{p.get('console_port', 5554)}":
        raise ValueError("serial must match the explicitly selected emulator console_port")
