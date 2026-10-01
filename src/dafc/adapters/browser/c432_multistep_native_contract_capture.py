#!/usr/bin/env python3
"""Capture C430 multi-step states under real browser render contracts.

This remains development-only.  DOM information is used to execute oracle
clicks and record scoring boxes, never as input to the visual method.  Each
episode is replayed under the five fixed C401 full-viewport contracts, and one
observation is captured before every action.  No GUI model or selector runs.
"""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any

from miniwob.environment import MiniWoBEnvironment


from dafc.paths import PROJECT
SCRIPTS = PROJECT / "scripts"
INPUT = PROJECT / "validation" / "c430_multistep_transition_feasibility.json"
REFLOW_PILOT = PROJECT / "validation" / "c434_narrow_reflow_width_pilot.json"
OUT_DIR = PROJECT / "tmp" / "c432_multistep_native_contract_capture"
OUTPUT = PROJECT / "validation" / "c432_multistep_native_contract_capture.json"


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
C400 = import_module('dafc.adapters.browser.c400_final_browser_capture')
C401 = import_module('dafc.adapters.browser.c401_full_viewport_browser_contracts')
C430 = import_module('dafc.adapters.browser.c430_multistep_transition_feasibility')


def multistep_contracts(selected_reflow_width: int) -> tuple[dict[str, Any], ...]:
    contracts = []
    for base in C401.CONTRACTS:
        if base["name"] != "native_narrow_reflow":
            contracts.append(dict(base))
            continue
        revised = {
            **base,
            "name": f"native_narrow_reflow_{selected_reflow_width}",
            "script": (
                "const w=document.getElementById('wrap');"
                "const q=document.getElementById('query');"
                "const a=document.getElementById('area');"
                f"w.style.width='{selected_reflow_width}px';"
                f"q.style.width='{selected_reflow_width}px';"
                f"a.style.width='{selected_reflow_width}px';"
            ),
            "multistep_reflow_revision": (
                f"C434 executable-width-{selected_reflow_width}px-v1"
            ),
        }
        contracts.append(revised)
    return tuple(contracts)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def candidate_boxes(
    driver, descriptors: list[dict[str, Any]], capture_scale: float
) -> list[dict[str, Any]]:
    boxes = []
    for index, descriptor in enumerate(descriptors):
        try:
            rect = C430.resolve_rect(driver, descriptor)
        except RuntimeError as error:
            boxes.append(
                {
                    "candidate_step": index,
                    "descriptor": descriptor,
                    "resolved": False,
                    "reason": str(error),
                }
            )
            continue
        center_css = C430.center(rect)
        boxes.append(
            {
                "candidate_step": index,
                "descriptor": descriptor,
                "resolved": True,
                "rect_css": rect,
                "center_css": center_css,
                "rect_image": [value * capture_scale for value in rect],
                "center_image": [value * capture_scale for value in center_css],
            }
        )
    return boxes


def point_visible(point: list[float], image: dict[str, Any]) -> bool:
    return bool(
        0.0 <= point[0] < float(image["width"])
        and 0.0 <= point[1] < float(image["height"])
    )


def run_contract_episode(
    source_row: dict[str, Any], contract: dict[str, Any]
) -> dict[str, Any]:
    task = str(source_row["task"])
    seed = int(source_row["seed"])
    env = MiniWoBEnvironment(subdomain=task, action_space_config="all_supported")
    started = time.perf_counter()
    try:
        env.instance.driver.set_window_size(900, 1000)
        observation, _ = env.reset(seed=seed)
        C400.reset_harness(env.instance.driver)
        plan = C430.action_plan(task, observation)
        expected_labels = [
            transition["descriptor"]["label"] for transition in source_row["transitions"]
        ]
        actual_labels = [descriptor["label"] for descriptor in plan]
        if observation["utterance"] != source_row["utterance"]:
            raise RuntimeError(f"utterance changed for {task} seed {seed}")
        if actual_labels != expected_labels:
            raise RuntimeError(
                f"action plan changed for {task} seed {seed}: "
                f"{actual_labels} != {expected_labels}"
            )
        if contract["script"]:
            env.instance.driver.execute_script(contract["script"])

        episode_dir = OUT_DIR / task / str(seed) / str(contract["name"])
        states = []
        outcomes = []
        suffix = ".jpg" if contract["format"] == "jpeg" else ".png"
        nonterminal_candidates = plan[:-1]
        for step, descriptor in enumerate(plan):
            state_path = episode_dir / f"state_{step:02d}{suffix}"
            image = C388.capture(env.instance.driver, state_path, contract)
            boxes = candidate_boxes(
                env.instance.driver,
                nonterminal_candidates,
                float(contract["capture_scale"]),
            )
            current_rect = C430.resolve_rect(env.instance.driver, descriptor)
            current_point = C430.center(current_rect)
            current_point_image = [
                value * float(contract["capture_scale"])
                for value in current_point
            ]
            if not point_visible(current_point_image, image):
                raise RuntimeError(
                    f"current action outside capture: {task} {seed} "
                    f"{contract['name']} step {step} {current_point_image}"
                )
            states.append(
                {
                    "state_index": step,
                    "next_action": descriptor,
                    "image": image,
                    "candidate_boxes_scoring_only": boxes,
                    "next_action_rect_css_scoring_only": current_rect,
                    "next_action_center_css_scoring_only": current_point,
                    "next_action_center_image_scoring_only": current_point_image,
                }
            )
            _, outcome = C430.click(env, current_point)
            C430.wait_for_visual_settle(env.instance.driver)
            outcomes.append(outcome)
            if outcome["terminated"] and step != len(plan) - 1:
                break

        final = outcomes[-1] if outcomes else None
        success = bool(
            final
            and float(final.get("raw_reward") or 0.0) >= 0.999
            and final.get("terminated")
            and len(outcomes) == len(plan)
        )
        return {
            "task": task,
            "seed": seed,
            "contract": contract["name"],
            "expected_global_support": bool(contract["supported"]),
            "exact_global_contract": bool(contract["exact_global"]),
            "capture_scale": float(contract["capture_scale"]),
            "planned_steps": len(plan),
            "executed_steps": len(outcomes),
            "success": success,
            "states": states,
            "outcomes": outcomes,
            "elapsed_s": time.perf_counter() - started,
        }
    finally:
        env.close()


def main() -> int:
    source = json.loads(INPUT.read_text(encoding="utf-8"))
    reflow_pilot = json.loads(REFLOW_PILOT.read_text(encoding="utf-8"))
    if source.get("paper_evidence") is not False or not source.get("pass"):
        raise RuntimeError("C430 must be a passing development-only record")
    if reflow_pilot.get("paper_evidence") is not False or not reflow_pilot.get("pass"):
        raise RuntimeError("C434 must be a passing development-only record")
    selected_reflow_width = int(reflow_pilot["selected_width_css"])
    contracts = multistep_contracts(selected_reflow_width)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    episodes = []
    total = len(source["rows"]) * len(contracts)
    for source_row in source["rows"]:
        for contract in contracts:
            result = run_contract_episode(source_row, contract)
            episodes.append(result)
            print(
                f"captured {len(episodes)}/{total} {result['task']} "
                f"seed={result['seed']} contract={result['contract']} "
                f"steps={result['executed_steps']} success={result['success']} "
                f"elapsed={result['elapsed_s']:.2f}s",
                flush=True,
            )

    images = [state["image"] for episode in episodes for state in episode["states"]]
    expected_states = sum(
        int(row["executed_steps"]) for row in source["rows"]
    ) * len(contracts)
    criteria = {
        "development_only": True,
        "zero_model_calls": True,
        "zero_selector_calls": True,
        "reflow_width_selected_by_c434": selected_reflow_width == 124,
        "exact_one_revised_reflow_contract": sum(
            contract["name"] == f"native_narrow_reflow_{selected_reflow_width}"
            for contract in contracts
        )
        == 1,
        "exact_120_contract_episodes": len(episodes) == 120,
        "exact_345_pre_action_states": len(images) == expected_states == 345,
        "all_contract_episodes_succeed": all(row["success"] for row in episodes),
        "all_current_actions_visible": all(
            point_visible(
                state["next_action_center_image_scoring_only"], state["image"]
            )
            for episode in episodes
            for state in episode["states"]
        ),
        "all_images_exist_and_hash_verify": all(
            (PROJECT / image["path"]).is_file()
            and sha256(PROJECT / image["path"]) == image["sha256"]
            for image in images
        ),
    }
    result = {
        "title": "C432 Multi-Step Native Browser Contract Capture",
        "paper_evidence": False,
        "purpose": (
            "development capture for estimated-geometry transition "
            "precondition testing"
        ),
        "inputs": [
            INPUT.relative_to(PROJECT).as_posix(),
            REFLOW_PILOT.relative_to(PROJECT).as_posix(),
        ],
        "contracts": [
            {
                "name": contract["name"],
                "expected_global_support": contract["supported"],
                "exact_global_contract": contract["exact_global"],
                "capture_scale": contract["capture_scale"],
                "format": contract["format"],
                "capture_revision": contract["capture_revision"],
                "multistep_reflow_revision": contract.get(
                    "multistep_reflow_revision"
                ),
            }
            for contract in contracts
        ],
        "summary": {
            "contract_episodes": len(episodes),
            "successful_contract_episodes": sum(
                int(row["success"]) for row in episodes
            ),
            "pre_action_states": len(images),
            "unique_image_hashes": len({image["sha256"] for image in images}),
            "image_bytes": sum(
                (PROJECT / image["path"]).stat().st_size for image in images
            ),
        },
        "criteria": criteria,
        "pass": all(criteria.values()),
        "episodes": episodes,
    }
    OUTPUT.write_text(
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
