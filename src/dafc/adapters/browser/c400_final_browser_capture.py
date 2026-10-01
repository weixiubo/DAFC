#!/usr/bin/env python3
"""Capture and verify all C398 final live-browser source/target observations."""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import hashlib
import importlib.util
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from miniwob.action import ActionTypes
from miniwob.environment import MiniWoBEnvironment


from dafc.paths import PROJECT
SCRIPTS = PROJECT / "scripts"
PREREGISTRATION = PROJECT / "validation" / "c398_browser_native_v6_preregistration.json"
QUALIFICATION = PROJECT / "validation" / "c399_final_browser_source_qualification.json"
OUT_DIR = PROJECT / "tmp" / "c400_final_browser_capture"
OUT_JSON = PROJECT / "validation" / "c400_final_browser_capture.json"


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
C388 = import_module('dafc.adapters.browser.c388_browser_native_contract_development_pilot')
C398 = import_module('dafc.adapters.browser.c398_browser_native_v6_preregister')


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return path.relative_to(PROJECT).as_posix()


def git_blob_matches_head(path: Path) -> bool:
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", rel(path)],
        cwd=PROJECT,
        capture_output=True,
        check=False,
    )
    if tracked.returncode != 0:
        return False
    clean = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", rel(path)],
        cwd=PROJECT,
        check=False,
    )
    return clean.returncode == 0


def reset_harness(driver) -> None:
    C388.reset_visual_styles(driver)
    driver.execute_script(
        "const c=document.getElementById('click-canvas');"
        "if(c){c.style.pointerEvents='none';}"
    )


def center(box: list[float]) -> list[float]:
    left, top, right, bottom = [float(value) for value in box]
    return [(left + right) / 2.0, (top + bottom) / 2.0]


def click_css_point(
    env: MiniWoBEnvironment, css_point: list[float]
) -> dict[str, Any]:
    action = env.create_action(
        ActionTypes.CLICK_COORDS,
        coords=np.asarray(css_point, dtype=np.float32),
    )
    _, reward, terminated, truncated, info = env.step(action)
    raw_reward = info.get("raw_reward")
    if hasattr(raw_reward, "item"):
        raw_reward = raw_reward.item()
    return {
        "css_point": [float(value) for value in css_point],
        "reward": float(reward),
        "raw_reward": raw_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "done": bool(info.get("done")),
    }


def successful_click(outcome: dict[str, Any]) -> bool:
    return bool(
        float(outcome.get("raw_reward") or 0.0) >= 0.999
        and outcome.get("terminated")
    )


def visible_center(box: list[float], width: int, height: int) -> bool:
    x, y = center(box)
    return 0.0 <= x < width and 0.0 <= y < height


def capture_source(source: dict[str, Any]) -> dict[str, Any]:
    task = source["task"]
    seed = int(source["seed"])
    target_box_css = [float(value) for value in source["target"]["box"]]
    source_id = f"{task}-{seed}"
    source_dir = OUT_DIR / source_id
    env = MiniWoBEnvironment(subdomain=task, action_space_config="all_supported")
    started = time.perf_counter()
    try:
        env.instance.driver.set_window_size(900, 1000)
        observation, _ = env.reset(seed=seed)
        if observation["utterance"] != source["utterance"]:
            raise RuntimeError(f"source utterance changed: {source_id}")
        reset_harness(env.instance.driver)
        source_locator = C388.mark_target_from_source_box(
            env.instance.driver, target_box_css
        )
        source_meta = C388.capture(
            env.instance.driver,
            source_dir / "source.png",
            C388.source_contract(),
        )
        source_box = C388.marked_target_rect(
            env.instance.driver, C388.SOURCE_SCALE
        )
        expected_source_box = [
            value * C388.SOURCE_SCALE for value in target_box_css
        ]
        if np.max(
            np.abs(np.asarray(source_box) - np.asarray(expected_source_box))
        ) > 0.1:
            raise RuntimeError(f"source target box changed: {source_id}")
        source_click = click_css_point(
            env,
            [value / C388.SOURCE_SCALE for value in center(source_box)],
        )

        targets = []
        for contract in C388.CONTRACTS:
            observation, _ = env.reset(seed=seed)
            if observation["utterance"] != source["utterance"]:
                raise RuntimeError(
                    f"contract utterance changed: {source_id} {contract['name']}"
                )
            reset_harness(env.instance.driver)
            target_locator = C388.mark_target_from_source_box(
                env.instance.driver,
                target_box_css,
                source_locator,
            )
            if contract["script"]:
                env.instance.driver.execute_script(contract["script"])
            suffix = ".jpg" if contract["format"] == "jpeg" else ".png"
            target_meta = C388.capture(
                env.instance.driver,
                source_dir / f"{contract['name']}{suffix}",
                contract,
            )
            target_box = C388.marked_target_rect(
                env.instance.driver, float(contract["capture_scale"])
            )
            target_center_visible = visible_center(
                target_box,
                int(target_meta["width"]),
                int(target_meta["height"]),
            )
            oracle_click = click_css_point(
                env,
                [
                    value / float(contract["capture_scale"])
                    for value in center(target_box)
                ],
            )
            targets.append(
                {
                    "contract": contract["name"],
                    "expected_global_support": contract["supported"],
                    "exact_global_contract": contract["exact_global"],
                    "capture_scale": float(contract["capture_scale"]),
                    "image": target_meta,
                    "scoring_target_bbox": target_box,
                    "scoring_target_center_visible": target_center_visible,
                    "scoring_oracle_center_click": oracle_click,
                    "target_locator_before_contract": target_locator,
                }
            )
        return {
            "source_id": source_id,
            "task": task,
            "seed": seed,
            "candidate_ordinal": int(source["candidate_ordinal"]),
            "utterance": source["utterance"],
            "fields": source["fields"],
            "source_image": source_meta,
            "source_target_bbox": source_box,
            "source_target_locator": source_locator,
            "source_oracle_center_click": source_click,
            "targets": targets,
            "elapsed_s": time.perf_counter() - started,
        }
    finally:
        env.close()


def main() -> int:
    if not git_blob_matches_head(Path(__file__).resolve()):
        raise RuntimeError("C400 script must be committed before capture")
    preregistration = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    qualification = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    if preregistration.get("pass") is not True or qualification.get("pass") is not True:
        raise RuntimeError("C398/C399 did not authorize final capture")
    if not git_blob_matches_head(PREREGISTRATION) or not git_blob_matches_head(
        QUALIFICATION
    ):
        raise RuntimeError("C398/C399 inputs must be committed")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sources = []
    for index, source in enumerate(qualification["selected_sources"], 1):
        captured = capture_source(source)
        sources.append(captured)
        print(
            f"captured {index}/80 {captured['source_id']} "
            f"elapsed={captured['elapsed_s']:.2f}s",
            flush=True,
        )
    targets = [target for source in sources for target in source["targets"]]
    source_hashes = [source["source_image"]["sha256"] for source in sources]
    target_hashes = [target["image"]["sha256"] for target in targets]
    criteria = {
        "exact_eighty_sources": len(sources) == 80,
        "exact_eight_sources_per_task": all(
            sum(source["task"] == task for source in sources) == 8
            for task in C398.TASKS
        ),
        "exact_four_hundred_targets": len(targets) == 400,
        "exact_five_targets_per_source": all(
            len(source["targets"]) == 5 for source in sources
        ),
        "all_source_oracle_clicks_succeed": all(
            successful_click(source["source_oracle_center_click"])
            for source in sources
        ),
        "all_target_oracle_clicks_succeed": all(
            successful_click(target["scoring_oracle_center_click"])
            for target in targets
        ),
        "all_target_centers_visible_in_captures": all(
            target["scoring_target_center_visible"] for target in targets
        ),
        "all_image_files_exist_and_hash_verify": all(
            sha256(PROJECT / source["source_image"]["path"])
            == source["source_image"]["sha256"]
            for source in sources
        )
        and all(
            sha256(PROJECT / target["image"]["path"])
            == target["image"]["sha256"]
            for target in targets
        ),
        "zero_model_calls": True,
        "zero_selector_calls": True,
    }
    result = {
        "title": "C400 Final Browser Observation Capture",
        "paper_evidence": True,
        "model_and_selector_blind_capture": True,
        "preregistration": {
            "path": rel(PREREGISTRATION),
            "sha256": sha256(PREREGISTRATION),
        },
        "qualification": {
            "path": rel(QUALIFICATION),
            "sha256": sha256(QUALIFICATION),
        },
        "script": {
            "path": rel(Path(__file__).resolve()),
            "sha256": sha256(Path(__file__).resolve()),
        },
        "harness_correction": {
            "reward_display_hidden": True,
            "click_canvas_pointer_events_none": True,
        },
        "summary": {
            "sources": len(sources),
            "unique_source_image_hashes": len(set(source_hashes)),
            "targets": len(targets),
            "unique_target_image_hashes": len(set(target_hashes)),
            "total_image_bytes": sum(
                (PROJECT / source["source_image"]["path"]).stat().st_size
                for source in sources
            )
            + sum(
                (PROJECT / target["image"]["path"]).stat().st_size
                for target in targets
            ),
        },
        "criteria": criteria,
        "pass": all(criteria.values()),
        "sources": sources,
    }
    OUT_JSON.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "summary": result["summary"],
                "criteria": criteria,
                "pass": result["pass"],
            },
            indent=2,
        )
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
