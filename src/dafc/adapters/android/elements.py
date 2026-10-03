from __future__ import annotations

import time
from typing import Any


def label(element: Any) -> str | None:
    return element.content_description or element.text


def click_element(env: Any, state: Any, target: str) -> dict[str, Any]:
    from android_world.env import json_action

    candidates = [
        (index, element)
        for index, element in enumerate(state.ui_elements)
        if target in {element.text, element.content_description}
        and element.bbox_pixels is not None
        and (element.is_visible is not False)
        and (element.is_enabled is not False)
    ]
    if not candidates:
        observed = sorted({value for element in state.ui_elements if (value := label(element))})
        raise RuntimeError(f"missing visible target {target!r}; observed={observed}")
    candidates.sort(key=lambda row: (row[1].is_clickable is not True, row[1].bbox_pixels.area, row[0]))
    index, element = candidates[0]
    x, y = element.bbox_pixels.center
    action = json_action.JSONAction(action_type="click", x=int(x), y=int(y))
    env.execute_action(action)
    time.sleep(2.0)
    return {
        "target_scoring_only": target,
        "ui_index_scoring_only": index,
        "coordinate_scoring_only": [int(x), int(y)],
        "bbox_scoring_only": [
            int(element.bbox_pixels.x_min),
            int(element.bbox_pixels.y_min),
            int(element.bbox_pixels.x_max),
            int(element.bbox_pixels.y_max),
        ],
    }
