from __future__ import annotations

import ast
import json
import math
import re
from typing import Any


def first_dictionary(text: str) -> dict[str, Any] | None:
    for candidate in re.findall("\\{[^{}]*\\}", text):
        try:
            parsed = ast.literal_eval(candidate)
        except Exception:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_legacy_click(text: str, kind: str, width: int, height: int) -> dict[str, Any]:
    raw = text.strip()
    dictionary = first_dictionary(raw)
    if dictionary is not None:
        action = str(dictionary.get("action", "")).upper()
        position = dictionary.get("position")
        if action in {"TAP", "CLICK"} and isinstance(position, (list, tuple)) and (len(position) >= 2):
            x, y = (float(position[0]), float(position[1]))
            x, y = _normalize_pair(x, y, kind, width, height)
            valid = bool(math.isfinite(x) and math.isfinite(y) and (0 <= x <= 1) and (0 <= y <= 1))
            return {"valid": valid, "parser": "action_dictionary", "normalized_xy": [x, y] if valid else None}
    patterns = (
        "<point>\\s*([-+]?\\d*\\.?\\d+)\\s+([-+]?\\d*\\.?\\d+)\\s*</point>",
        "<\\|box_start\\|>\\s*\\(\\s*([-+]?\\d*\\.?\\d+)\\s*,\\s*([-+]?\\d*\\.?\\d+)\\s*\\)\\s*<\\|box_end\\|>",
        "click\\s*\\([^\\d+-]*([-+]?\\d*\\.?\\d+)\\s*[, ]+\\s*([-+]?\\d*\\.?\\d+)",
        "\\[\\s*([-+]?\\d*\\.?\\d+)\\s*,\\s*([-+]?\\d*\\.?\\d+)\\s*\\]",
    )
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match is None:
            continue
        x, y = (float(match.group(1)), float(match.group(2)))
        x, y = _normalize_pair(x, y, kind, width, height)
        valid = bool(math.isfinite(x) and math.isfinite(y) and (0 <= x <= 1) and (0 <= y <= 1))
        return {"valid": valid, "parser": "coordinate_regex", "normalized_xy": [x, y] if valid else None}
    return {"valid": False, "parser": "parse_failure", "normalized_xy": None}


def _normalize_pair(x: float, y: float, kind: str, width: int, height: int) -> tuple[float, float]:
    if max(abs(x), abs(y)) > 1.0:
        return (x / width, y / height) if kind == "qwen25vl" else (x / 1000.0, y / 1000.0)
    return (x, y)
