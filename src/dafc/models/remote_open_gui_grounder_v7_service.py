#!/usr/bin/env python3
"""Persistent Qwen service for the V7 smart-resized model-frame interface."""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import sys

import dafc.models.remote_open_gui_grounder_v5_service as base
import dafc.models.remote_open_gui_grounder_v7_qwen_model_frame as runner


SERVICE_VERSION = "remote-open-gui-grounder-v7-qwen-model-frame-service"


def main() -> int:
    if "--kind" not in sys.argv:
        raise ValueError("--kind qwen25vl is required")
    kind_index = sys.argv.index("--kind") + 1
    if kind_index >= len(sys.argv) or sys.argv[kind_index] != "qwen25vl":
        raise ValueError("V7 model-frame service supports only --kind qwen25vl")
    base.v3 = runner
    base.SERVICE_VERSION = SERVICE_VERSION
    base.BASE_RUNNER_VERSION = runner.RUNNER_VERSION
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
