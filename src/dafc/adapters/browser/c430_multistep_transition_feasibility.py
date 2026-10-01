#!/usr/bin/env python3
"""Development-only feasibility study for visual transition memory.

This script uses DOM-derived oracle actions only to characterize task mechanics
before a new method or confirmatory protocol is frozen.  It asks whether
natural multi-step MiniWoB tasks expose a visible pre/post condition for each
point click.  No row is paper evidence and no model is called.
"""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import importlib.util
import json
import re
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from miniwob.action import ActionTypes
from miniwob.environment import MiniWoBEnvironment


from dafc.paths import PROJECT
SCRIPTS = PROJECT / "scripts"
OUT_DIR = PROJECT / "tmp" / "c430_multistep_transition_feasibility"
OUT_JSON = PROJECT / "validation" / "c430_multistep_transition_feasibility.json"

TASKS = (
    "click-button-sequence",
    "click-checkboxes",
    "click-checkboxes-large",
    "click-checkboxes-transfer",
    "click-collapsible",
    "click-collapsible-nodelay",
    "click-collapsible-2",
    "click-collapsible-2-nodelay",
)
SEEDS = (430001, 430002, 430003)


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


def target_fields(observation: dict[str, Any]) -> list[str]:
    return [
        str(value)
        for key, value in observation["fields"]
        if str(key).startswith("target")
    ]


def quoted_target(utterance: str) -> str:
    match = re.search(r'link\s+"([^"]+)"', utterance)
    if not match:
        raise RuntimeError(f"could not parse link target: {utterance}")
    return match.group(1)


def action_plan(task: str, observation: dict[str, Any]) -> list[dict[str, Any]]:
    if task == "click-button-sequence":
        return [
            {"kind": "selector", "selector": "#subbtn", "label": "button ONE"},
            {"kind": "selector", "selector": "#subbtn2", "label": "button TWO"},
        ]
    if task.startswith("click-checkboxes"):
        return [
            {"kind": "checkbox_text", "text": text, "label": f"checkbox {text}"}
            for text in target_fields(observation)
        ] + [{"kind": "selector", "selector": "#subbtn", "label": "Submit"}]
    if task in {"click-collapsible", "click-collapsible-nodelay"}:
        return [
            {"kind": "selector", "selector": "#area h3", "label": "expand"},
            {"kind": "selector", "selector": "#subbtn", "label": "Submit"},
        ]
    if task in {"click-collapsible-2", "click-collapsible-2-nodelay"}:
        target = quoted_target(observation["utterance"])
        return [
            {"kind": "accordion_for_link", "text": target, "label": "expand target section"},
            {"kind": "link_text", "text": target, "label": f"link {target}"},
        ]
    raise KeyError(task)


def resolve_rect(driver, action: dict[str, Any]) -> list[float]:
    kind = action["kind"]
    if kind == "selector":
        result = driver.execute_script(
            "const e=document.querySelector(arguments[0]);"
            "if(!e)return null;const r=e.getBoundingClientRect();"
            "return [r.left,r.top,r.right,r.bottom];",
            action["selector"],
        )
    elif kind == "checkbox_text":
        result = driver.execute_script(
            "const wanted=arguments[0];"
            "for(const label of document.querySelectorAll('label')){"
            " if((label.textContent||'').trim()===wanted){"
            "  const e=label.querySelector('input[type=checkbox]');"
            "  if(!e)continue;const r=e.getBoundingClientRect();"
            "  return [r.left,r.top,r.right,r.bottom];}}return null;",
            action["text"],
        )
    elif kind == "link_text":
        result = driver.execute_script(
            "const wanted=arguments[0];"
            "for(const e of document.querySelectorAll('.alink')){"
            " if((e.textContent||'').trim()===wanted){"
            "  const r=e.getBoundingClientRect();"
            "  return [r.left,r.top,r.right,r.bottom];}}return null;",
            action["text"],
        )
    elif kind == "accordion_for_link":
        result = driver.execute_script(
            "const wanted=arguments[0];"
            "for(const e of document.querySelectorAll('.alink')){"
            " if((e.textContent||'').trim()!==wanted)continue;"
            " const panel=e.closest('.ui-accordion-content');"
            " if(!panel)return null;let h=panel.previousElementSibling;"
            " while(h&&h.tagName.toLowerCase()!=='h3')h=h.previousElementSibling;"
            " if(!h)return null;const r=h.getBoundingClientRect();"
            " return [r.left,r.top,r.right,r.bottom];}return null;",
            action["text"],
        )
    else:
        raise KeyError(kind)
    if result is None:
        raise RuntimeError(f"could not resolve action: {action}")
    return [float(value) for value in result]


def center(rect: list[float]) -> list[float]:
    return [(rect[0] + rect[2]) / 2.0, (rect[1] + rect[3]) / 2.0]


def click(env: MiniWoBEnvironment, point: list[float]) -> tuple[dict[str, Any], dict[str, Any]]:
    action = env.create_action(
        ActionTypes.CLICK_COORDS, coords=np.asarray(point, dtype=np.float32)
    )
    observation, reward, terminated, truncated, info = env.step(action)
    raw_reward = info.get("raw_reward")
    if hasattr(raw_reward, "item"):
        raw_reward = raw_reward.item()
    return observation, {
        "reward": float(reward),
        "raw_reward": raw_reward,
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "done": bool(info.get("done")),
    }


def effect_metrics(before_path: Path, after_path: Path) -> dict[str, Any]:
    before = cv2.imread(str(before_path), cv2.IMREAD_GRAYSCALE)
    after = cv2.imread(str(after_path), cv2.IMREAD_GRAYSCALE)
    if before is None or after is None or before.shape != after.shape:
        raise RuntimeError(f"invalid effect images: {before_path} {after_path}")
    delta = cv2.absdiff(before, after)
    changed = delta > 12
    ys, xs = np.where(changed)
    bbox = None
    if len(xs):
        bbox = [int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)]
    fraction = float(changed.mean())
    mae = float(delta.mean())
    return {
        "grayscale_mae": mae,
        "changed_gt12_fraction": fraction,
        "changed_gt12_pixels": int(changed.sum()),
        "effect_bbox": bbox,
        "visible_effect": bool(mae >= 0.10 or fraction >= 0.0005),
    }


def capture_state(driver, path: Path) -> dict[str, Any]:
    return C388.capture(driver, path, C388.source_contract())


def wait_for_visual_settle(driver, timeout_s: float = 2.5) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        animating = driver.execute_script(
            "return window.jQuery ? jQuery(':animated').length : 0;"
        )
        if int(animating or 0) == 0:
            return
        time.sleep(0.05)
    raise RuntimeError("browser animation did not settle")


def run_episode(task: str, seed: int) -> dict[str, Any]:
    env = MiniWoBEnvironment(subdomain=task, action_space_config="all_supported")
    episode_dir = OUT_DIR / task / str(seed)
    try:
        env.instance.driver.set_window_size(900, 1000)
        observation, _ = env.reset(seed=seed)
        C400.reset_harness(env.instance.driver)
        initial_utterance = observation["utterance"]
        initial_fields = [list(item) for item in observation["fields"]]
        plan = action_plan(task, observation)
        transitions = []
        final_outcome: dict[str, Any] | None = None
        for index, descriptor in enumerate(plan):
            before_path = episode_dir / f"step_{index:02d}_before.png"
            after_path = episode_dir / f"step_{index:02d}_after.png"
            before = capture_state(env.instance.driver, before_path)
            rect = resolve_rect(env.instance.driver, descriptor)
            point = center(rect)
            if not (0.0 <= point[0] < 160.0 and 0.0 <= point[1] < 210.0):
                raise RuntimeError(f"oracle point outside task canvas: {task} {seed} {point}")
            observation, outcome = click(env, point)
            wait_for_visual_settle(env.instance.driver)
            after = capture_state(env.instance.driver, after_path)
            effect = effect_metrics(before_path, after_path)
            transitions.append(
                {
                    "step": index,
                    "descriptor": descriptor,
                    "resolved_rect_css": rect,
                    "action_css": point,
                    "before": before,
                    "after": after,
                    "effect": effect,
                    "outcome": outcome,
                }
            )
            final_outcome = outcome
            if outcome["terminated"] and index != len(plan) - 1:
                break
        success = bool(
            final_outcome
            and float(final_outcome.get("raw_reward") or 0.0) >= 0.999
            and final_outcome.get("terminated")
            and len(transitions) == len(plan)
        )
        return {
            "task": task,
            "seed": seed,
            "utterance": initial_utterance,
            "fields": initial_fields,
            "planned_steps": len(plan),
            "executed_steps": len(transitions),
            "visible_effect_steps": sum(
                int(row["effect"]["visible_effect"]) for row in transitions
            ),
            "success": success,
            "transitions": transitions,
        }
    finally:
        env.close()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [run_episode(task, seed) for task in TASKS for seed in SEEDS]
    nonterminal = [
        transition
        for row in rows
        for transition in row["transitions"][:-1]
    ]
    task_summary = []
    for task in TASKS:
        task_rows = [row for row in rows if row["task"] == task]
        task_summary.append(
            {
                "task": task,
                "episodes": len(task_rows),
                "successful_episodes": sum(int(row["success"]) for row in task_rows),
                "multistep_episodes": sum(
                    int(row["executed_steps"] >= 2) for row in task_rows
                ),
                "mean_steps": float(np.mean([row["executed_steps"] for row in task_rows])),
                "visible_effect_rate": float(
                    np.mean(
                        [
                            transition["effect"]["visible_effect"]
                            for row in task_rows
                            for transition in row["transitions"][:-1]
                        ]
                    )
                ),
            }
        )
    criteria = {
        "development_only": True,
        "zero_model_calls": True,
        "exact_eight_tasks": len(TASKS) == 8,
        "exact_three_development_seeds": len(SEEDS) == 3,
        "all_24_oracle_episodes_succeed": all(row["success"] for row in rows),
        "at_least_20_episodes_require_multiple_clicks": sum(
            int(row["executed_steps"] >= 2) for row in rows
        )
        >= 20,
        "at_least_six_tasks_have_visible_nonterminal_effects": sum(
            int(row["visible_effect_rate"] > 0.0) for row in task_summary
        )
        >= 6,
    }
    result = {
        "title": "C430 Multi-Step Visual Transition Feasibility",
        "paper_evidence": False,
        "purpose": "task mechanics and visible pre/post condition development only",
        "tasks": list(TASKS),
        "seeds": list(SEEDS),
        "summary": {
            "episodes": len(rows),
            "successful_episodes": sum(int(row["success"]) for row in rows),
            "transitions": sum(row["executed_steps"] for row in rows),
            "nonterminal_transitions": len(nonterminal),
            "visible_nonterminal_effect_rate": float(
                np.mean([row["effect"]["visible_effect"] for row in nonterminal])
            ),
        },
        "task_summary": task_summary,
        "criteria": criteria,
        "pass": all(criteria.values()),
        "rows": rows,
    }
    OUT_JSON.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "summary": result["summary"],
                "task_summary": task_summary,
                "criteria": criteria,
                "pass": result["pass"],
            },
            indent=2,
        )
    )
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
