from __future__ import annotations

import base64
import hashlib
import io
from pathlib import Path
from typing import Any

from PIL import Image

CSS_WIDTH = 160.0
CSS_HEIGHT = 210.0
SOURCE_SCALE = 3.0
CONTRACTS = (
    {
        "name": "native_zoom_125",
        "supported": True,
        "exact_global": False,
        "capture_scale": 3.0,
        "clip_width": CSS_WIDTH * 1.25,
        "clip_height": CSS_HEIGHT * 1.25,
        "format": "png",
        "script": "document.getElementById('wrap').style.zoom='1.25';",
        "matrix": [[1.25, 0.0, 0.0], [0.0, 1.25, 0.0], [0.0, 0.0, 1.0]],
    },
    {
        "name": "native_zoom_80",
        "supported": True,
        "exact_global": False,
        "capture_scale": 3.0,
        "clip_width": CSS_WIDTH * 0.8,
        "clip_height": CSS_HEIGHT * 0.8,
        "format": "png",
        "script": "document.getElementById('wrap').style.zoom='0.8';",
        "matrix": [[0.8, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 1.0]],
    },
    {
        "name": "native_affine_skew",
        "supported": True,
        "exact_global": True,
        "capture_scale": 3.0,
        "clip_width": 200.0,
        "clip_height": 235.0,
        "format": "png",
        "script": "const w=document.getElementById('wrap');w.style.transformOrigin='0 0';w.style.transform='matrix(1.02,0.08,-0.05,0.96,18,8)';",
        "matrix": [[1.02, -0.05, 54.0], [0.08, 0.96, 24.0], [0.0, 0.0, 1.0]],
    },
    {
        "name": "native_jpeg_stream",
        "supported": True,
        "exact_global": True,
        "capture_scale": 2.0,
        "clip_width": CSS_WIDTH,
        "clip_height": CSS_HEIGHT,
        "format": "jpeg",
        "quality": 82,
        "script": "",
        "matrix": [[2.0 / 3.0, 0.0, 0.0], [0.0, 2.0 / 3.0, 0.0], [0.0, 0.0, 1.0]],
    },
    {
        "name": "native_narrow_reflow",
        "supported": False,
        "exact_global": False,
        "capture_scale": 3.0,
        "clip_width": CSS_WIDTH,
        "clip_height": CSS_HEIGHT,
        "format": "png",
        "script": "const w=document.getElementById('wrap');const q=document.getElementById('query');const a=document.getElementById('area');w.style.width='108px';q.style.width='108px';a.style.width='108px';",
        "matrix": None,
    },
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(driver, path: Path, contract: dict[str, Any]) -> dict[str, Any]:
    params: dict[str, Any] = {
        "format": contract["format"],
        "fromSurface": True,
        "captureBeyondViewport": False,
        "clip": {
            "x": 0.0,
            "y": 0.0,
            "width": float(contract["clip_width"]),
            "height": float(contract["clip_height"]),
            "scale": float(contract["capture_scale"]),
        },
    }
    if contract["format"] == "jpeg":
        params["quality"] = int(contract["quality"])
    payload = driver.execute_cdp_cmd("Page.captureScreenshot", params)
    data = base64.b64decode(payload["data"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    image = Image.open(io.BytesIO(data)).convert("RGB")
    return {
        "path": artifact_reference(path),
        "sha256": hashlib.sha256(data).hexdigest(),
        "width": image.width,
        "height": image.height,
        "format": contract["format"],
    }


def source_contract() -> dict[str, Any]:
    return {
        "name": "source",
        "capture_scale": SOURCE_SCALE,
        "clip_width": CSS_WIDTH,
        "clip_height": CSS_HEIGHT,
        "format": "png",
    }


def reset_visual_styles(driver) -> None:
    driver.execute_script(
        "const w=document.getElementById('wrap');const q=document.getElementById('query');const a=document.getElementById('area');w.style.zoom='';w.style.transform='';w.style.transformOrigin='';w.style.width='';q.style.width='';a.style.width='';const r=document.getElementById('reward-display');if(r){r.style.display='none';}"
    )


def artifact_reference(path: Path) -> str:
    return str(path.resolve())
