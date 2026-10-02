"""Explicit browser driver binding and the fixed 600-second episode timer."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
from pathlib import Path
from threading import RLock
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from miniwob.environment import MiniWoBEnvironment
HARNESS_VERSION = "miniwob-online-agent-harness-v2-explicit-driver"
AGENT_EPISODE_MAX_TIME_MS = 600000
EXPECTED_CHROMEDRIVER_SHA256 = "51df048171813deb307f7bd26baf49db030032f9332cedce8ce2797e03b86be1"
_DRIVER_BINDING_LOCK = RLock()


def verify_driver(driver: Path, expected_sha256: str) -> None:
    if not driver.is_file():
        raise ValueError(f"ChromeDriver missing: {driver}")
    if sha256(driver) != expected_sha256:
        raise ValueError(f"ChromeDriver checksum mismatch: {driver}")


@contextmanager
def driver_binding(driver: Path, expected_sha256: str):
    """MiniWoB 1.0 has no Service injection API; bound and restored under a lock."""
    import miniwob.selenium_instance as instance
    from selenium.webdriver.chrome.service import Service

    verify_driver(driver, expected_sha256)
    with _DRIVER_BINDING_LOCK:
        original = instance.webdriver.Chrome

        def bound_chrome(*args, **kwargs):
            if "service" in kwargs:
                raise ValueError("MiniWoB unexpectedly supplied a Chrome service")
            return original(*args, service=Service(executable_path=str(driver)), **kwargs)

        instance.webdriver.Chrome = bound_chrome
        try:
            yield
        finally:
            instance.webdriver.Chrome = original


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def reset_agent_episode(env: MiniWoBEnvironment, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
    env.instance.driver.execute_script("core.EPISODE_MAX_TIME=arguments[0];", AGENT_EPISODE_MAX_TIME_MS)
    observation, info = env.reset(seed=seed)
    observed = int(env.instance.driver.execute_script("return core.EPISODE_MAX_TIME;"))
    if observed != AGENT_EPISODE_MAX_TIME_MS:
        raise RuntimeError(f"agent episode timer changed: {observed} != {AGENT_EPISODE_MAX_TIME_MS}")
    return (observation, info)
