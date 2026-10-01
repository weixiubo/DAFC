#!/usr/bin/env python3
"""Development pilot for browser-native MiniWoB visual action contracts.

This uses one qualified development seed per task family to choose viable
browser-native contracts before the external-validity protocol is frozen.
Neither these seeds nor any tuned contract parameter may enter final evidence.
"""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import base64
import hashlib
import importlib.util
import io
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
from miniwob.environment import MiniWoBEnvironment


from dafc.paths import PROJECT
SCRIPTS = PROJECT / "scripts"
QUALIFICATION = PROJECT / "validation" / "c387_miniwob_single_click_qualification_pilot.json"
OUT_DIR = PROJECT / "tmp" / "c388_browser_native_contract_development_pilot"
OUT_JSON = PROJECT / "validation" / "c388_browser_native_contract_development_pilot.json"

CSS_WIDTH = 160.0
CSS_HEIGHT = 210.0
SOURCE_SCALE = 3.0


def artifact_reference(path: Path) -> str:
    """Serialize repository artifacts relatively and formal artifacts absolutely."""

    resolved = path.resolve()
    try:
        return resolved.relative_to(PROJECT.resolve()).as_posix()
    except ValueError:
        return str(resolved)


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
V5 = import_module('dafc.registration.selector_v5')


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
        "clip_width": CSS_WIDTH * 0.80,
        "clip_height": CSS_HEIGHT * 0.80,
        "format": "png",
        "script": "document.getElementById('wrap').style.zoom='0.8';",
        "matrix": [[0.80, 0.0, 0.0], [0.0, 0.80, 0.0], [0.0, 0.0, 1.0]],
    },
    {
        "name": "native_affine_skew",
        "supported": True,
        "exact_global": True,
        "capture_scale": 3.0,
        "clip_width": 200.0,
        "clip_height": 235.0,
        "format": "png",
        "script": (
            "const w=document.getElementById('wrap');"
            "w.style.transformOrigin='0 0';"
            "w.style.transform='matrix(1.02,0.08,-0.05,0.96,18,8)';"
        ),
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
        "matrix": [
            [2.0 / 3.0, 0.0, 0.0],
            [0.0, 2.0 / 3.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
    },
    {
        "name": "native_narrow_reflow",
        "supported": False,
        "exact_global": False,
        "capture_scale": 3.0,
        "clip_width": CSS_WIDTH,
        "clip_height": CSS_HEIGHT,
        "format": "png",
        "script": (
            "const w=document.getElementById('wrap');"
            "const q=document.getElementById('query');"
            "const a=document.getElementById('area');"
            "w.style.width='108px';q.style.width='108px';a.style.width='108px';"
        ),
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
        "const w=document.getElementById('wrap');"
        "const q=document.getElementById('query');"
        "const a=document.getElementById('area');"
        "w.style.zoom='';w.style.transform='';w.style.transformOrigin='';"
        "w.style.width='';q.style.width='';a.style.width='';"
        "const r=document.getElementById('reward-display');"
        "if(r){r.style.display='none';}"
    )


def mark_target_from_source_box(
    driver,
    source_box_css: list[float],
    source_signature: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the qualified element before styling and give it a stable marker.

    MiniWoB's ``data-wob_ref`` values are episode-local and can refer to a
    different element after ``reset``.  The same-seed unstyled layout is
    deterministic, however, so the scoring element can be recovered from the
    exact clipped box identified by C387 and marked before a contract changes
    layout.  The marker is scoring-only and is never exposed to the selector.
    """

    result = driver.execute_script(
        "const expected=arguments[0].map(Number);"
        "const signature=arguments[1];"
        "const marker='data-c388-scoring-target';"
        "for(const old of document.querySelectorAll('['+marker+']')){"
        " old.removeAttribute(marker); }"
        "const rows=[];"
        "for(const e of document.querySelectorAll('[data-wob_ref]')){"
        " const r=e.getBoundingClientRect();"
        " const clipped=[Math.max(0,r.left),Math.max(0,r.top),"
        "  Math.min(160,r.right),Math.min(210,r.bottom)];"
        " if(clipped[2]<=clipped[0]||clipped[3]<=clipped[1]) continue;"
        " const error=Math.max(...clipped.map((v,i)=>Math.abs(v-expected[i])));"
        " const attrs={tag:e.tagName.toLowerCase(),id:e.id||'',"
        "  classes:String(e.className||''),text:(e.innerText||'').trim(),"
        "  value:e.value||''};"
        " let mismatches=0;"
        " if(signature){for(const k of ['tag','id','classes','text','value']){"
        "  if(String(attrs[k])!==String(signature[k]||'')) mismatches++;}}"
        " rows.push({e,r,clipped,error,area:Math.max(0,r.width*r.height),"
        "  attrs,mismatches});"
        "}"
        "rows.sort((a,b)=>a.mismatches-b.mismatches||a.error-b.error||a.area-b.area);"
        "if(!rows.length||(rows[0].mismatches>0&&rows[0].error>0.35)){"
        " return {ok:false,best_error:rows.length?rows[0].error:null,"
        "  best_mismatches:rows.length?rows[0].mismatches:null}; }"
        "const hit=rows[0];hit.e.setAttribute(marker,'1');"
        "return {ok:true,best_error:hit.error,signature_mismatches:hit.mismatches,"
        " ref:hit.e.getAttribute('data-wob_ref'),"
        " tag:hit.attrs.tag,id:hit.attrs.id,classes:hit.attrs.classes,"
        " text:hit.attrs.text,value:hit.attrs.value,"
        " rect:[hit.r.left,hit.r.top,hit.r.right,hit.r.bottom],"
        " clipped_rect:hit.clipped};",
        source_box_css,
        source_signature,
    )
    if not result or not result.get("ok"):
        raise RuntimeError(
            "could not resolve qualified target from unstyled source box "
            f"{source_box_css}: {result}"
        )
    return result


def marked_target_rect(driver, scale: float) -> list[float]:
    rect = driver.execute_script(
        "const e=document.querySelector('[data-c388-scoring-target=\"1\"]');"
        "if(!e){return null;} const r=e.getBoundingClientRect();"
        "return [r.left,r.top,r.right,r.bottom];"
    )
    if rect is None:
        raise RuntimeError("marked scoring target missing after transform")
    return [float(item) * scale for item in rect]


def map_points(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.concatenate(
        [points.astype(np.float64), np.ones((len(points), 1), dtype=np.float64)],
        axis=1,
    )
    mapped = homogeneous @ matrix.T
    return mapped[:, :2] / mapped[:, 2:3]


def grid_error(estimated: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    xs = np.linspace(0.0, CSS_WIDTH * SOURCE_SCALE - 1.0, 17)
    ys = np.linspace(0.0, CSS_HEIGHT * SOURCE_SCALE - 1.0, 17)
    points = np.array([(x, y) for y in ys for x in xs], dtype=np.float64)
    error = np.linalg.norm(
        map_points(estimated, points) - map_points(truth, points), axis=1
    )
    return {
        "median": float(np.median(error)),
        "p95": float(np.percentile(error, 95)),
        "maximum": float(np.max(error)),
    }


def box_points(box: list[float]) -> tuple[np.ndarray, np.ndarray]:
    left, top, right, bottom = box
    corners = np.array(
        [[left, top], [right, top], [right, bottom], [left, bottom]],
        dtype=np.float64,
    )
    grid = np.array(
        [
            [left + fx * (right - left), top + fy * (bottom - top)]
            for fy in (0.1, 0.5, 0.9)
            for fx in (0.1, 0.5, 0.9)
        ],
        dtype=np.float64,
    )
    return corners, grid


def action_audit(
    estimated: np.ndarray,
    source_box: list[float],
    observed_target_box: list[float],
) -> dict[str, Any]:
    _, grid = box_points(source_box)
    target_corners, _ = box_points(observed_target_box)
    target_polygon = target_corners.astype(np.float32)
    estimated_grid = map_points(estimated, grid)
    inside = [
        cv2.pointPolygonTest(target_polygon, tuple(point.astype(float)), False) >= 0
        for point in estimated_grid
    ]
    target_center = np.asarray(
        [
            (observed_target_box[0] + observed_target_box[2]) / 2.0,
            (observed_target_box[1] + observed_target_box[3]) / 2.0,
        ],
        dtype=np.float64,
    )
    return {
        "all_core_points_inside": all(inside),
        "inside_count": sum(int(item) for item in inside),
        "points": len(inside),
        "mapped_center": estimated_grid[4].tolist(),
        "observed_target_center": target_center.tolist(),
        "center_to_target_center_px": float(
            np.linalg.norm(estimated_grid[4] - target_center)
        ),
    }


def development_sources() -> list[dict[str, Any]]:
    data = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    sources = []
    for task_row in data["task_rows"]:
        if not task_row["qualifies"]:
            continue
        row = next(item for item in task_row["rows"] if item["qualifies"])
        sources.append(
            {
                "task": task_row["task"],
                "seed": row["seed"],
                "utterance": row["utterance"],
                "target_ref": row["selected_target"]["ref"],
                "target_box_css": row["selected_target"]["box"],
            }
        )
    return sources


def run_source(source: dict[str, Any]) -> list[dict[str, Any]]:
    task = source["task"]
    seed = source["seed"]
    task_dir = OUT_DIR / f"{task}_{seed}"
    env = MiniWoBEnvironment(subdomain=task, action_space_config="all_supported")
    try:
        env.instance.driver.set_window_size(900, 1000)
        observation, _ = env.reset(seed=seed)
        if observation["utterance"] != source["utterance"]:
            raise RuntimeError(f"source utterance changed: {task} {seed}")
        reset_visual_styles(env.instance.driver)
        source_locator = mark_target_from_source_box(
            env.instance.driver, source["target_box_css"]
        )
        source_meta = capture(
            env.instance.driver, task_dir / "source.png", source_contract()
        )
        source_box = marked_target_rect(env.instance.driver, SOURCE_SCALE)
        qualified_box = [
            item * SOURCE_SCALE for item in source["target_box_css"]
        ]
        if np.max(np.abs(np.asarray(source_box) - np.asarray(qualified_box))) > 0.1:
            raise RuntimeError(f"qualified target box changed: {task} {seed}")
        rows = []
        for contract in CONTRACTS:
            observation, _ = env.reset(seed=seed)
            if observation["utterance"] != source["utterance"]:
                raise RuntimeError(f"contract utterance changed: {task} {seed}")
            reset_visual_styles(env.instance.driver)
            target_locator = mark_target_from_source_box(
                env.instance.driver,
                source["target_box_css"],
                source_locator,
            )
            if contract["script"]:
                env.instance.driver.execute_script(contract["script"])
            target_path = task_dir / (
                contract["name"] + (".jpg" if contract["format"] == "jpeg" else ".png")
            )
            target_meta = capture(env.instance.driver, target_path, contract)
            observed_target_box = marked_target_rect(
                env.instance.driver, float(contract["capture_scale"])
            )
            source_image = cv2.imread(str(PROJECT / source_meta["path"]), cv2.IMREAD_COLOR)
            target_image = cv2.imread(str(PROJECT / target_meta["path"]), cv2.IMREAD_COLOR)
            if source_image is None or target_image is None:
                raise RuntimeError("OpenCV decode failed")
            started = time.perf_counter()
            selector = V5.register_contract(source_image, target_image)
            runtime = time.perf_counter() - started
            row: dict[str, Any] = {
                "task": task,
                "seed": seed,
                "utterance": source["utterance"],
                "contract": contract["name"],
                "expected_supported": contract["supported"],
                "exact_global_contract": contract["exact_global"],
                "source": source_meta,
                "target": target_meta,
                "source_target_box": source_box,
                "observed_target_bbox": observed_target_box,
                "source_target_locator": source_locator,
                "target_locator_before_contract": target_locator,
                "selector_supported": bool(selector.get("supported")),
                "selector_reason": selector.get("reason"),
                "selector_family": selector.get("selected_family"),
                "selector_decision_path": selector.get("decision_path"),
                "runtime_s": runtime,
            }
            if contract["supported"] and selector.get("supported"):
                estimated = np.asarray(selector["matrix"], dtype=np.float64)
                if contract["exact_global"]:
                    truth = np.asarray(contract["matrix"], dtype=np.float64)
                    row["grid_error_px"] = grid_error(estimated, truth)
                row["action_audit"] = action_audit(
                    estimated, source_box, observed_target_box
                )
            rows.append(row)
        return rows
    finally:
        env.close()


def main() -> int:
    cv2.setNumThreads(1)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = development_sources()
    rows = [row for source in sources for row in run_source(source)]
    supported = [row for row in rows if row["expected_supported"]]
    unsupported = [row for row in rows if not row["expected_supported"]]
    accepted = [row for row in supported if row["selector_supported"]]
    bad = [
        row
        for row in accepted
        if row.get("grid_error_px", {}).get("p95", 0.0) >= 3.0
    ]
    unsafe = [
        row
        for row in accepted
        if not row["action_audit"]["all_core_points_inside"]
    ]
    by_contract = {}
    for contract in CONTRACTS:
        contract_rows = [row for row in rows if row["contract"] == contract["name"]]
        by_contract[contract["name"]] = {
            "expected_supported": contract["supported"],
            "exact_global_contract": contract["exact_global"],
            "total": len(contract_rows),
            "accepted": sum(int(row["selector_supported"]) for row in contract_rows),
            "rejected": sum(int(not row["selector_supported"]) for row in contract_rows),
        }
    result = {
        "title": "Browser-Native Contract Development Pilot",
        "paper_evidence": False,
        "source_count": len(sources),
        "pair_count": len(rows),
        "supported": {
            "total": len(supported),
            "accepted": len(accepted),
            "bad_accepted_maps": len(bad),
            "unsafe_accepted_action_cores": len(unsafe),
        },
        "unsupported": {
            "total": len(unsupported),
            "rejected": sum(int(not row["selector_supported"]) for row in unsupported),
        },
        "runtime_s": {
            "mean": float(np.mean([row["runtime_s"] for row in rows])),
            "p95": float(np.percentile([row["runtime_s"] for row in rows], 95)),
        },
        "by_contract": by_contract,
        "rows": rows,
    }
    result["pass"] = bool(
        len(accepted) >= math.ceil(0.80 * len(supported))
        and not bad
        and not unsafe
        and result["unsupported"]["rejected"] >= math.ceil(0.80 * len(unsupported))
    )
    OUT_JSON.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {key: result[key] for key in result if key not in {"rows"}},
            indent=2,
        )
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
