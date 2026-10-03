"""Fixed seven-task atomic study design; ordering and cohort unchanged."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from dafc.routing import POLICIES

NAMESPACE = "tg-aaai27-c587-android-atomic-v3"
ACTORS = ("qwen25vl", "uitars")
CONTRACTS = ("identity", "jpeg-70", "density-480", "font-1.15")
REPETITIONS = 5


@dataclass(frozen=True)
class AtomicTask:
    task_id: str
    params: dict[str, int | str]
    app_name: str
    setup_prefix_labels: tuple[str, ...]
    terminal_label_scoring_only: str


TASKS = (
    AtomicTask("ClockStopWatchRunning", {}, "clock", ("Stopwatch",), "Start"),
    AtomicTask(
        "ClockTimerEntry", {"hours": 0, "minutes": 1, "seconds": 30}, "clock", ("Timer", "1", "3"), "0"
    ),
    AtomicTask(
        "SystemWifiTurnOn", {"on_or_off": "on"}, "settings", ("Network & internet", "Internet"), "Wi-Fi"
    ),
    AtomicTask(
        "SystemBluetoothTurnOn",
        {"on_or_off": "on"},
        "settings",
        ("Connected devices", "Connection preferences", "Bluetooth"),
        "Use Bluetooth",
    ),
    AtomicTask(
        "SystemWifiTurnOff", {"on_or_off": "off"}, "settings", ("Network & internet", "Internet"), "Wi-Fi"
    ),
    AtomicTask(
        "SystemBluetoothTurnOff",
        {"on_or_off": "off"},
        "settings",
        ("Connected devices", "Connection preferences", "Bluetooth"),
        "Use Bluetooth",
    ),
    AtomicTask("AudioRecorderRecordAudio", {}, "audio recorder", (), "Recording: %s"),
)
TASK_BY_ID = {row.task_id: row for row in TASKS}


def job_key(actor: str, task: str, contract: str, policy: str, repetition: int) -> str:
    return f"{actor}|{task}|{contract}|{policy}|r{repetition:02d}"


def job_id(actor: str, task: str, contract: str, policy: str, repetition: int) -> str:
    return f"{NAMESPACE}--{actor}--{task}--{contract}--{policy}--r{repetition:02d}"


def jobs_for_actor(actor: str) -> list[dict[str, Any]]:
    if actor not in ACTORS:
        raise ValueError(f"unknown actor: {actor}")
    jobs = [
        {
            "actor": actor,
            "task": task.task_id,
            "contract": contract,
            "policy": policy,
            "repetition": repetition,
            "job_id": job_id(actor, task.task_id, contract, policy, repetition),
        }
        for task in TASKS
        for contract in CONTRACTS
        for policy in POLICIES
        for repetition in range(REPETITIONS)
    ]
    return sorted(
        jobs,
        key=lambda row: hashlib.sha256(
            f"{NAMESPACE}|{job_key(row['actor'], row['task'], row['contract'], row['policy'], row['repetition'])}".encode(
                "utf-8"
            )
        ).hexdigest(),
    )
