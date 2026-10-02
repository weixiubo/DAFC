"""Qwen GUI grounder with an explicit smart-resized model-frame interface."""

from __future__ import annotations

import math
import re
from typing import Any

import dafc.models.grounding as base

RUNNER_VERSION = "remote-open-gui-grounder-v7-qwen-model-frame"
PROCESSOR_USE_FAST = base.PROCESSOR_USE_FAST
PROCESSOR_PADDING_SIDE = base.PROCESSOR_PADDING_SIDE
MAX_NEW_TOKENS = base.MAX_NEW_TOKENS
MIN_PIXELS = base.MIN_PIXELS
MAX_PIXELS = base.MAX_PIXELS
SMART_RESIZE_FACTOR = 28
MAX_ASPECT_RATIO = 200
QWEN_MODEL_FRAME_PROMPT = "You are a deterministic GUI grounding agent.\nFind the requested target in the full screenshot and output exactly one click.\nReturn one line in this form and no explanation:\nAction: click(point='[x,y]')\nThe visual processor presents the full screenshot as a model frame that is\nexactly {model_width} pixels wide and {model_height} pixels high. x and y must\nbe integer pixel coordinates in that model frame: 0 <= x <= {model_width} and\n0 <= y <= {model_height}. Do not normalize, do not use the original screenshot\ndimensions, and do not return a bounding box.\nTask: {instruction}\n"
checkpoint_fingerprint = base.checkpoint_fingerprint


def _round_by_factor(value: float, factor: int) -> int:
    return round(value / factor) * factor


def _ceil_by_factor(value: float, factor: int) -> int:
    return math.ceil(value / factor) * factor


def _floor_by_factor(value: float, factor: int) -> int:
    return math.floor(value / factor) * factor


def smart_resize_model_frame(width: int, height: int) -> tuple[int, int]:
    """Return Qwen's processor frame size as ``(width, height)``.

    This mirrors the Qwen-VL smart-resize rule for the processor bounds frozen
    in ``grounding``.  The explicit argument and return order
    prevents the height/width swap that is easy to make with the upstream
    helper.
    """
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if max(width, height) / min(width, height) > MAX_ASPECT_RATIO:
        raise ValueError("image aspect ratio exceeds Qwen's smart-resize bound")
    resized_height = _round_by_factor(height, SMART_RESIZE_FACTOR)
    resized_width = _round_by_factor(width, SMART_RESIZE_FACTOR)
    resized_pixels = resized_height * resized_width
    if resized_pixels > MAX_PIXELS:
        beta = math.sqrt(height * width / MAX_PIXELS)
        resized_height = max(SMART_RESIZE_FACTOR, _floor_by_factor(height / beta, SMART_RESIZE_FACTOR))
        resized_width = max(SMART_RESIZE_FACTOR, _floor_by_factor(width / beta, SMART_RESIZE_FACTOR))
    elif resized_pixels < MIN_PIXELS:
        beta = math.sqrt(MIN_PIXELS / (height * width))
        resized_height = _ceil_by_factor(height * beta, SMART_RESIZE_FACTOR)
        resized_width = _ceil_by_factor(width * beta, SMART_RESIZE_FACTOR)
    if resized_width <= 0 or resized_height <= 0:
        raise ValueError("smart resize produced a non-positive model frame")
    return (resized_width, resized_height)


def qwen_model_frame_prompt(kind: str, row: dict[str, Any], width: int, height: int) -> str:
    if kind != "qwen25vl":
        raise ValueError("V7 model-frame prompt supports only qwen25vl")
    model_width, model_height = smart_resize_model_frame(width, height)
    return QWEN_MODEL_FRAME_PROMPT.format(
        instruction=str(row["instruction"]), model_width=model_width, model_height=model_height
    )


def parse_model_frame_click(text: str, kind: str, width: int, height: int) -> dict[str, Any]:
    if kind != "qwen25vl":
        raise ValueError("V7 model-frame parser supports only qwen25vl")
    model_width, model_height = smart_resize_model_frame(width, height)
    raw = text.strip()
    patterns = (
        "click\\s*\\([^\\d+-]*([-+]?\\d*\\.?\\d+)\\s*[, ]+\\s*([-+]?\\d*\\.?\\d+)",
        "\\[\\s*([-+]?\\d*\\.?\\d+)\\s*,\\s*([-+]?\\d*\\.?\\d+)\\s*\\]",
    )
    match = next(
        (
            candidate
            for pattern in patterns
            if (candidate := re.search(pattern, raw, flags=re.IGNORECASE)) is not None
        ),
        None,
    )
    common = {
        "coordinate_scale": "qwen_smart_resized_model_frame_pixels",
        "model_frame_width": model_width,
        "model_frame_height": model_height,
    }
    if match is None:
        return {
            **common,
            "valid": False,
            "parser": "model_frame_parse_failure",
            "raw_xy": None,
            "normalized_xy": None,
        }
    x, y = (float(match.group(1)), float(match.group(2)))
    valid = bool(
        math.isfinite(x) and math.isfinite(y) and (0 <= x <= model_width) and (0 <= y <= model_height)
    )
    return {
        **common,
        "valid": valid,
        "parser": "model_frame_coordinate_regex",
        "raw_xy": [x, y],
        "normalized_xy": [x / model_width, y / model_height] if valid else None,
    }


def load_model(kind, model_path):
    if kind != "qwen25vl":
        raise ValueError("model-frame backend supports only qwen25vl")
    return base.load_model(kind, model_path)


def infer_one(model, processor, kind: str, row: dict[str, Any]) -> dict[str, Any]:
    if kind != "qwen25vl":
        raise ValueError("V7 model-frame service supports only qwen25vl")
    result = base.infer_one(
        model,
        processor,
        kind,
        row,
        prompt_builder=qwen_model_frame_prompt,
        action_parser=parse_model_frame_click,
    )
    model_width, model_height = smart_resize_model_frame(result["image_width"], result["image_height"])
    result["coordinate_interface"] = {
        "version": RUNNER_VERSION,
        "scale": "qwen_smart_resized_model_frame_pixels",
        "model_frame_width": model_width,
        "model_frame_height": model_height,
        "executor_maps_to_original_screenshot": True,
    }
    return result


if __name__ == "__main__":
    raise SystemExit("Use model_frame_service.py for persistent inference")
