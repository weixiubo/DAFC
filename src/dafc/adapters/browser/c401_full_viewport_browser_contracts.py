#!/usr/bin/env python3
"""Fixed full-viewport capture revision of the C388 browser contracts.

The browser styles, capture scales, formats, JPEG quality, and expected-map
labels are inherited unchanged.  Only the CDP clip is enlarged to a common
220x360 CSS viewport so every task target remains visible after native zoom or
responsive reflow.  C400's failed final states are the sole development set
for this capture-only revision.
"""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import importlib.util
import sys
from pathlib import Path


from dafc.paths import PROJECT
SCRIPTS = PROJECT / "scripts"
BASE_PATH = SCRIPTS / "c388_browser_native_contract_development_pilot.py"
FULL_VIEWPORT_CSS_WIDTH = 220.0
FULL_VIEWPORT_CSS_HEIGHT = 360.0


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
BASE = import_module('dafc.adapters.browser.c388_browser_native_contract_development_pilot')

CONTRACTS = tuple(
    {
        **contract,
        "clip_width": FULL_VIEWPORT_CSS_WIDTH,
        "clip_height": FULL_VIEWPORT_CSS_HEIGHT,
        "capture_revision": "fixed-full-viewport-220x360-v1",
    }
    for contract in BASE.CONTRACTS
)


def main() -> int:
    print(
        {
            "capture_revision": "fixed-full-viewport-220x360-v1",
            "clip_width_css": FULL_VIEWPORT_CSS_WIDTH,
            "clip_height_css": FULL_VIEWPORT_CSS_HEIGHT,
            "contracts": [contract["name"] for contract in CONTRACTS],
        }
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
