"""Strict portable run configuration and explicit input bindings."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from dafc.resources import ResourcePaths, resolve_paths, sha256_file

WORKFLOWS = {"routing", "browser-episode", "android-episode", "android-atomic"}


@dataclass(frozen=True)
class InputFile:
    path: Path
    sha256: str | None


@dataclass(frozen=True)
class RunConfig:
    workflow: str
    run_id: str
    output: Path
    resources: ResourcePaths
    inputs: dict[str, InputFile]
    parameters: dict[str, Any]
    services: dict[str, str]
    resume: bool
    config_file: Path


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or any((not isinstance(key, str) for key in value)):
        raise ValueError(f"{label} must be a mapping with string keys")
    return value


def _inputs(values: Any, paths: ResourcePaths, base: Path) -> dict[str, InputFile]:
    result = {}
    for name, entry in _mapping(values, "inputs").items():
        item = _mapping(entry, f"inputs.{name}")
        if set(item) - {"path", "sha256"} or not isinstance(item.get("path"), str):
            raise ValueError(f"inputs.{name} requires path and optional sha256")
        digest = item.get("sha256")
        if digest is not None and (
            not isinstance(digest, str) or re.fullmatch("[0-9a-f]{64}", digest) is None
        ):
            raise ValueError(f"inputs.{name}.sha256 must be lowercase SHA-256")
        result[name] = InputFile(paths.expand(item["path"], base=base), digest)
    return result


def load_config(path: Path, overrides: Mapping[str, Path | None] | None = None) -> RunConfig:
    path = path.expanduser().resolve()
    payload = _mapping(yaml.safe_load(path.read_text()), "configuration")
    unknown = set(payload) - {
        "workflow",
        "run_id",
        "output",
        "paths",
        "inputs",
        "parameters",
        "services",
        "resume",
    }
    if unknown:
        raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
    workflow = payload.get("workflow")
    if not isinstance(workflow, str) or workflow not in WORKFLOWS:
        raise ValueError(f"workflow must be one of {sorted(WORKFLOWS)}")
    run_id = payload.get("run_id")
    if not isinstance(run_id, str) or re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]{0,79}", run_id) is None:
        raise ValueError("run_id must be a safe 1-80 character name")
    paths = _configured_paths(payload, path, overrides)
    services = _mapping(payload.get("services", {}), "services")
    if any(
        (
            not isinstance(value, str) or not value.startswith(("http://", "https://"))
            for value in services.values()
        )
    ):
        raise ValueError("services must contain explicit HTTP(S) URLs")
    if not isinstance(payload.get("resume", False), bool):
        raise ValueError("resume must be true or false")
    output_value = payload.get("output", "{runs}/" + run_id)
    if not isinstance(output_value, str) or not output_value:
        raise ValueError("output must be a nonempty path string")
    output = paths.expand(output_value, base=path.parent)
    return RunConfig(
        workflow,
        run_id,
        output,
        paths,
        _inputs(payload.get("inputs", {}), paths, path.parent),
        _mapping(payload.get("parameters", {}), "parameters"),
        services,
        payload.get("resume", False),
        path,
    )


def input_problems(config: RunConfig) -> list[str]:
    problems = []
    for name, item in config.inputs.items():
        if not item.path.is_file():
            problems.append(
                f"Input {name} is missing or not a file: {item.path}; restore it and update inputs.{name}.path"
            )
        elif item.sha256 is not None and sha256_file(item.path) != item.sha256:
            problems.append(
                f"Input {name} checksum mismatch: {item.path}; restore the version matching inputs.{name}.sha256"
            )
    return problems


def _configured_paths(
    payload: dict[str, Any], path: Path, overrides: Mapping[str, Path | None] | None
) -> ResourcePaths:
    configured_paths = _mapping(payload.get("paths", {}), "paths")
    if set(configured_paths) - {"data", "models", "runs", "evidence"}:
        raise ValueError("paths supports data, models, runs and evidence")
    if any((not isinstance(value, str) or not value for value in configured_paths.values())):
        raise ValueError("paths values must be nonempty path strings")
    explicit = {
        key: (path.parent / Path(value).expanduser()).resolve() for key, value in configured_paths.items()
    }
    explicit.update({key: value for key, value in (overrides or {}).items() if value is not None})
    paths = resolve_paths(explicit)
    return paths
