"""Validate source outcome and immutable image receipts before a new experiment."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

from dafc.paths import receipt_path
from dafc.storage import load_json, sha256


def validate_source(row: dict[str, Any], *, resource_base: Path) -> dict[str, Any]:
    artifact = receipt_path(row["artifact"], base=resource_base)
    if not artifact.is_file() or sha256(artifact) != row["artifact_sha256"]:
        raise RuntimeError(f"source artifact hash mismatch: {artifact}")
    episode = load_json(artifact)
    _validate_episode_identity(row, episode)
    summary = {
        "task": row["task"],
        "goal": row["goal"],
        "params": row["params"],
        "source_success": bool(row["source_success"]),
        "artifact": str(artifact),
        "artifact_sha256": row["artifact_sha256"],
        "episode_id": row["episode_id"],
        "source_model_calls": int(row["model_calls"]),
        "source_official_reward": float(row["official_reward"]),
    }
    if not row["source_success"]:
        if episode.get("success") or float(episode.get("final_reward", -1)) != 0.0:
            raise RuntimeError("failed source unexpectedly succeeded")
        return {**summary, "atomic_memory": None, "source_failure_reason": episode.get("failure_reason")}
    return {
        **summary,
        "atomic_memory": _terminal_memory(episode, resource_base),
        "source_failure_reason": None,
    }


def _validate_episode_identity(row, episode):
    if (
        episode.get("task_id") != row["task"]
        or episode.get("goal") != row["goal"]
        or episode.get("params") != row["params"]
        or (bool(episode.get("success")) != bool(row["source_success"]))
        or (float(episode.get("final_reward", -1)) != float(row["official_reward"]))
        or (episode.get("policy") != "always")
        or (episode.get("contract") != "identity")
    ):
        raise RuntimeError(f"source registry/episode mismatch: {row['task']}")


def _terminal_memory(episode, resource_base):
    executed = [step for step in episode["steps"] if step.get("action_executed")]
    rewards = [float(step["official_reward_after_action"]) for step in executed]
    if (
        not executed
        or rewards[-1] != 1.0
        or any((value != 0.0 for value in rewards[:-1]))
        or (float(episode.get("initial_reward", -1)) != 0.0)
    ):
        raise RuntimeError("successful source lacks a unique terminal action receipt")
    terminal = executed[-1]
    before = receipt_path(terminal["before"]["path"], base=resource_base)
    after = receipt_path(terminal["after"]["path"], base=resource_base)
    for path, receipt in ((before, terminal["before"]), (after, terminal["after"])):
        if not path.is_file() or sha256(path) != receipt["sha256"]:
            raise RuntimeError(f"source transition image mismatch: {path}")
    action = _action_point(terminal)
    return {
        "source_before": str(before),
        "source_before_sha256": terminal["before"]["sha256"],
        "source_after": str(after),
        "source_after_sha256": terminal["after"]["sha256"],
        "source_action_xy": action,
        "source_step": int(terminal["step"]),
        "official_reward_after_action": 1.0,
        "earlier_executed_actions": len(executed) - 1,
        "earlier_rewards_all_zero": True,
    }


def _action_point(terminal):
    action = [float(value) for value in terminal["action_xy"]]
    if len(action) != 2 or not all((math.isfinite(value) for value in action)):
        raise RuntimeError("source terminal action is not a finite point")
    return action
