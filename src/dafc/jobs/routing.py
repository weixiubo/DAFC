"""One offline image-to-image routing decision."""

from __future__ import annotations

import cv2

from dafc.configuration import RunConfig
from dafc.resources import sha256_file
from dafc.routing import authorize


def run_routing(config: RunConfig) -> dict:
    source = config.inputs["source"].path
    target = cv2.imread(str(config.inputs["target"].path), cv2.IMREAD_COLOR)
    receipt = authorize(
        config.parameters.get("policy", "dafc"),
        {
            "source_success": config.parameters["source_success"],
            "atomic_memory": {
                "source_before": str(source),
                "source_before_sha256": sha256_file(source),
                "source_action_xy": config.parameters["action_xy"],
            },
        },
        target,
    )
    return {"workflow": "routing", "run_id": config.run_id, "decision": receipt, "paper_evidence": False}
