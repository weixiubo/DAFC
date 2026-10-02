from __future__ import annotations

from typing import Any

from dafc.adapters.browser import capture

FULL_VIEWPORT_CSS_WIDTH = 220.0
FULL_VIEWPORT_CSS_HEIGHT = 360.0
FULL_VIEWPORT_CONTRACTS = tuple(
    (
        {
            **contract,
            "clip_width": FULL_VIEWPORT_CSS_WIDTH,
            "clip_height": FULL_VIEWPORT_CSS_HEIGHT,
            "capture_revision": "fixed-full-viewport-220x360-v1",
        }
        for contract in capture.CONTRACTS
    )
)


def multistep_contracts(selected_reflow_width: int) -> tuple[dict[str, Any], ...]:
    contracts = []
    for base in FULL_VIEWPORT_CONTRACTS:
        if base["name"] != "native_narrow_reflow":
            contracts.append(dict(base))
            continue
        revised = {
            **base,
            "name": f"native_narrow_reflow_{selected_reflow_width}",
            "script": f"const w=document.getElementById('wrap');const q=document.getElementById('query');const a=document.getElementById('area');w.style.width='{selected_reflow_width}px';q.style.width='{selected_reflow_width}px';a.style.width='{selected_reflow_width}px';",
            "multistep_reflow_revision": f"C434 executable-width-{selected_reflow_width}px-v1",
        }
        contracts.append(revised)
    return tuple(contracts)
