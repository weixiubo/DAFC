#!/usr/bin/env python3
"""Reproducible MiniWoB browser harness for online GUI agents.

MiniWoB's bundled page uses a 10-second demonstration timer.  That timer makes
correctness depend on model and registration wall time, so online agent runs
set a 600-second ceiling before each reset and report wall-clock latency as a
separate endpoint.  Raw task reward remains unchanged.  The harness also binds
the locally cached ChromeDriver explicitly to avoid Selenium Manager races.
"""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import hashlib
from pathlib import Path
from typing import Any

import miniwob.selenium_instance as miniwob_selenium_instance
from miniwob.environment import MiniWoBEnvironment
from selenium import webdriver as selenium_webdriver
from selenium.webdriver.chrome.service import Service


HARNESS_VERSION = "miniwob-online-agent-harness-v1"
AGENT_EPISODE_MAX_TIME_MS = 600_000
CHROMEDRIVER = (
    Path.home()
    / ".cache"
    / "selenium"
    / "chromedriver"
    / "linux64"
    / "146.0.7680.165"
    / "chromedriver"
)
EXPECTED_CHROMEDRIVER_SHA256 = (
    "51df048171813deb307f7bd26baf49db030032f9332cedce8ce2797e03b86be1"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_driver() -> None:
    if not CHROMEDRIVER.is_file():
        raise RuntimeError(f"bound ChromeDriver is missing: {CHROMEDRIVER}")
    observed = sha256(CHROMEDRIVER)
    if observed != EXPECTED_CHROMEDRIVER_SHA256:
        raise RuntimeError(
            f"ChromeDriver hash mismatch: {observed} != "
            f"{EXPECTED_CHROMEDRIVER_SHA256}"
        )


def bind_driver() -> None:
    verify_driver()
    current = miniwob_selenium_instance.webdriver.Chrome
    if getattr(current, "_transition_guard_bound_driver", False):
        return
    original = current

    def bound_chrome(*args, **kwargs):
        if "service" in kwargs:
            raise ValueError("MiniWoB unexpectedly supplied a Chrome service")
        return original(
            *args,
            service=Service(executable_path=str(CHROMEDRIVER)),
            **kwargs,
        )

    bound_chrome._transition_guard_bound_driver = True
    miniwob_selenium_instance.webdriver.Chrome = bound_chrome


def reset_agent_episode(
    env: MiniWoBEnvironment, seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    env.instance.driver.execute_script(
        "core.EPISODE_MAX_TIME=arguments[0];", AGENT_EPISODE_MAX_TIME_MS
    )
    observation, info = env.reset(seed=seed)
    observed = int(
        env.instance.driver.execute_script("return core.EPISODE_MAX_TIME;")
    )
    if observed != AGENT_EPISODE_MAX_TIME_MS:
        raise RuntimeError(
            f"agent episode timer changed: {observed} != "
            f"{AGENT_EPISODE_MAX_TIME_MS}"
        )
    return observation, info


def binding() -> dict[str, Any]:
    verify_driver()
    return {
        "harness_version": HARNESS_VERSION,
        "agent_episode_max_time_ms": AGENT_EPISODE_MAX_TIME_MS,
        "success_reward": "raw MiniWoB reward; wall-clock latency reported separately",
        "chromedriver": {
            "path": str(CHROMEDRIVER),
            "bytes": CHROMEDRIVER.stat().st_size,
            "sha256": sha256(CHROMEDRIVER),
        },
    }


bind_driver()
