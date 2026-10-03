from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from dafc.adapters.browser import capture


def reset_harness(driver) -> None:
    capture.reset_visual_styles(driver)
    driver.execute_script(
        "const c=document.getElementById('click-canvas');if(c){c.style.pointerEvents='none';}"
    )


def click_css_point(env: MiniWoBEnvironment, css_point: list[float]) -> dict[str, Any]:
    from miniwob.action import ActionTypes

    action = env.create_action(ActionTypes.CLICK_COORDS, coords=np.asarray(css_point, dtype=np.float32))
    _, reward, terminated, truncated, info = env.step(action)
    raw_reward = info.get("raw_reward")
    if hasattr(raw_reward, "item"):
        raw_reward = raw_reward.item()
    return {
        "css_point": [float(value) for value in css_point],
        "reward": float(reward),
        "raw_reward": raw_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "done": bool(info.get("done")),
    }


def successful_click(outcome: dict[str, Any]) -> bool:
    return bool(float(outcome.get("raw_reward") or 0.0) >= 0.999 and outcome.get("terminated"))


if TYPE_CHECKING:
    from miniwob.environment import MiniWoBEnvironment
