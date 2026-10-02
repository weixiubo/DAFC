from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

import numpy as np


def click(env: MiniWoBEnvironment, point: list[float]) -> tuple[dict[str, Any], dict[str, Any]]:
    from miniwob.action import ActionTypes

    action = env.create_action(ActionTypes.CLICK_COORDS, coords=np.asarray(point, dtype=np.float32))
    observation, reward, terminated, truncated, info = env.step(action)
    raw_reward = info.get("raw_reward")
    if hasattr(raw_reward, "item"):
        raw_reward = raw_reward.item()
    return (
        observation,
        {
            "reward": float(reward),
            "raw_reward": raw_reward,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "done": bool(info.get("done")),
        },
    )


def wait_for_visual_settle(driver, timeout_s: float = 2.5) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        animating = driver.execute_script("return window.jQuery ? jQuery(':animated').length : 0;")
        if int(animating or 0) == 0:
            return
        time.sleep(0.05)
    raise RuntimeError("browser animation did not settle")


if TYPE_CHECKING:
    from miniwob.environment import MiniWoBEnvironment
