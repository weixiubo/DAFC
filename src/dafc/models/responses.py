from __future__ import annotations
import hashlib, json, math, time
from pathlib import Path
from typing import Any
import cv2
import numpy as np


def target_instruction(goal: str) -> str:
    return (
        f"Original Android task: {goal}\n"
        "The benchmark harness has placed the required app at the final "
        "decision state. Using only the current screenshot, select exactly "
        "one executable tap that completes the task. Do not explain the answer."
    )

def compact_model_response(response: dict[str, Any]) -> dict[str, Any]:
    result = response.get("result") or {}
    return {
        "service_version": response.get("service_version"),
        "service_s": response.get("service_s"),
        "id": result.get("id"),
        "image_sha256": result.get("image_sha256"),
        "instruction": result.get("instruction"),
        "parsed_action": result.get("parsed_action"),
        "raw_response": result.get("raw_response"),
        "input_tokens": result.get("input_tokens"),
        "generated_tokens": result.get("generated_tokens"),
        "inference_s": result.get("inference_s"),
    }

def normalized_to_pixel(parsed: dict[str, Any], image: np.ndarray) -> list[float] | None:
    if not parsed.get("valid"):
        return None
    values = parsed.get("normalized_xy")
    if (
        not isinstance(values, list)
        or len(values) != 2
        or not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values)
        or not all(math.isfinite(float(v)) for v in values)
    ):
        return None
    height, width = image.shape[:2]
    return [
        min(max(float(values[0]) * width, 0.0), width - 1.0),
        min(max(float(values[1]) * height, 0.0), height - 1.0),
    ]
