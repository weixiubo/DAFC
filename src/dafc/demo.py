"""Small synthetic geometry demonstration, independent of model and platform services."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import cv2
import numpy as np

from dafc.routing import authorize
from dafc.storage import sha256


def demo() -> dict:
    rng = np.random.default_rng(585001)
    source_image = cv2.GaussianBlur(rng.integers(0, 256, (360, 480, 3), dtype=np.uint8), (3, 3), 0)
    target = cv2.warpAffine(source_image, np.float32([[1, 0, 12], [0, 1, 8]]), (480, 360))
    with TemporaryDirectory(prefix="dafc-demo-") as directory:
        source_file = Path(directory) / "source.png"
        if not cv2.imwrite(str(source_file), source_image):
            raise RuntimeError("Could not encode the synthetic source image")
        source = {
            "source_success": True,
            "atomic_memory": {
                "source_before": str(source_file),
                "source_before_sha256": sha256(source_file),
                "source_action_xy": [211.0, 143.0],
            },
        }
        reuse = authorize("dafc", source, target)
        fallback = authorize("dafc", {"source_success": False}, target)
    return {
        "status": "ok",
        "synthetic": True,
        "seed": 585001,
        "reuse_authorized": reuse["accepted"],
        "mapped_action_xy": reuse["mapped_action_xy"],
        "failed_source_falls_back": not fallback["accepted"],
        "model_calls": 0,
        "device_calls": 0,
    }
