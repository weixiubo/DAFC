"""History-aware, versioned batch runner for open GUI-agent checkpoints.

V2 is immutable because several completed preregistrations bind its hash.  V3
keeps the same dataset-agnostic manifest interface, freezes processor behavior,
and adds the official HAR-GUI history prompt and pixel-coordinate decoder.
DOM geometry, target annotations, and outcome scoring remain outside the
remote process.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

from dafc.models.legacy_coordinates import parse_legacy_click as parse_legacy_click

RUNNER_VERSION = "remote-open-gui-grounder-v3"
PROCESSOR_USE_FAST = True
PROCESSOR_PADDING_SIDE = "left"
MAX_NEW_TOKENS = 256
MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 1008 * 28 * 28
UITARS_PROMPT = "You are a GUI action agent.\nGiven the screenshot and task, output the next executable action.\n\nAction space:\n- click(point='<point>x y</point>'): click the target at integer coordinates x y in a 0-1000 screen coordinate system.\n\nReturn one line starting with Action: and no extra explanation.\nTask: {instruction}\n"
SHOWUI_PROMPT = "You are an assistant trained to navigate a GUI screen.\nGiven a task instruction and a screen observation, output the next action.\nReturn a dictionary exactly in this form:\n{{'action': 'TAP', 'value': None, 'position': [x,y]}}\nPosition is relative to the full screenshot and must be scaled to 0-1.\nTask: {instruction}\n"
QWEN_PROMPT = "You are a deterministic GUI grounding agent.\nFind the requested target in the full screenshot and output exactly one click.\nReturn one line in this form and no explanation:\nAction: click(point='[x,y]')\nThe full screenshot is {width} pixels wide and {height} pixels high. x and y\nmust be exact full-screenshot pixel coordinates: 0 <= x <= {width} and\n0 <= y <= {height}. Do not normalize and do not return a bounding box.\nTask: {instruction}\n"
HAR_ACTION_SPACE = '\nCLICK:(x,y): Click on the element at the coordinate point (x,y) on the screen, e.g., CLICK:(1980,224).\nTYPE:typed_text: Type text, e.g., TYPE:"Macbook-Pro 16G Black".\nCOMPLETE: The goal has been completed in the current screen state.\nSCROLL:UP/DOWN/LEFT/RIGHT: Scroll in a specific direction, e.g., SCROLL:UP.\nLONG_PRESS:(x,y): Long press at a point, e.g., LONG_PRESS:(345,2218).\nBACK: Go back to the previous screen.\nHOME: Go to the home screen.\n'.strip()
HAR_PROMPT = 'You are a skilled assistant, interacting with the screen to accomplish the user\'s goals.\nHere is the action space:\n{action_space}\nYour overall goal is: <goal>{goal}</goal>\nActions completed at previous steps: <history>{history}</history>\n\nThe output format should be as follows:\n<think>Analyze step by step based on guidance and screen state to choose the action.</think>\n<answer>The action you finally choose from "action space".</answer>'


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_fingerprint(path: Path) -> dict[str, Any]:
    config = path / "config.json"
    weights = sorted(
        (item for pattern in ("*.safetensors", "*.bin") for item in path.glob(pattern) if item.is_file())
    )
    return {
        "scheme": "dafc-checkpoint-v2",
        "path": str(path.resolve()),
        "config_sha256": sha256(config),
        "weight_files": [
            {"name": item.name, "bytes": item.stat().st_size, "sha256": sha256(item)} for item in weights
        ],
        "total_weight_bytes": sum((item.stat().st_size for item in weights)),
    }


def prompt(kind: str, row: dict[str, Any], width: int, height: int) -> str:
    if kind == "har":
        goal = str(row.get("goal", "")).strip()
        history = str(row.get("history", "")).strip()
        if not goal or not history:
            raise ValueError("HAR rows require non-empty goal and history fields")
        return HAR_PROMPT.format(action_space=HAR_ACTION_SPACE, goal=goal, history=history)
    instruction = str(row["instruction"])
    if kind == "uitars":
        return UITARS_PROMPT.format(instruction=instruction)
    if kind == "showui":
        return SHOWUI_PROMPT.format(instruction=instruction)
    if kind == "qwen25vl":
        return QWEN_PROMPT.format(instruction=instruction, width=width, height=height)
    raise ValueError(f"unknown model kind: {kind}")


def parse_har_click(text: str, width: int, height: int) -> dict[str, Any]:
    raw = text.strip()
    answer_match = re.search("<answer>(.*?)</answer>", raw, flags=re.IGNORECASE | re.DOTALL)
    answer = answer_match.group(1).strip() if answer_match else raw
    match = re.search(
        "CLICK\\s*:\\s*\\(\\s*([-+]?\\d*\\.?\\d+)\\s*,\\s*([-+]?\\d*\\.?\\d+)\\s*\\)",
        answer,
        flags=re.IGNORECASE,
    )
    if match is None:
        return {"valid": False, "parser": "har_click_parse_failure", "normalized_xy": None}
    x, y = (float(match.group(1)), float(match.group(2)))
    if max(abs(x), abs(y)) <= 1.0:
        normalized = [x, y]
        scale = "normalized"
    else:
        normalized = [x / width, y / height]
        scale = "full_screenshot_pixels"
    valid = bool(all((math.isfinite(value) and 0.0 <= value <= 1.0 for value in normalized)))
    return {
        "valid": valid,
        "parser": "har_answer_click",
        "coordinate_scale": scale,
        "raw_xy": [x, y],
        "normalized_xy": normalized if valid else None,
    }


def parse_normalized_click(text: str, kind: str, width: int, height: int) -> dict[str, Any]:
    if kind == "har":
        return parse_har_click(text, width, height)
    return parse_legacy_click(text, kind, width, height)


def load_model(kind: str, model_path: Path):
    import torch
    from transformers import (
        AutoProcessor,
        Qwen2_5_VLForConditionalGeneration,
        Qwen2VLForConditionalGeneration,
    )

    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        padding_side=PROCESSOR_PADDING_SIDE,
        use_fast=PROCESSOR_USE_FAST,
        local_files_only=True,
    )
    model_class = (
        Qwen2_5_VLForConditionalGeneration if kind in {"qwen25vl", "har"} else Qwen2VLForConditionalGeneration
    )
    model = model_class.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, device_map="auto", local_files_only=True
    )
    model.eval()
    return (model, processor)


def infer_one(
    model,
    processor,
    kind: str,
    row: dict[str, Any],
    *,
    prompt_builder=prompt,
    action_parser=parse_normalized_click,
) -> dict[str, Any]:
    import torch
    from PIL import Image

    image_path = Path(row["image_path"])
    image = Image.open(image_path).convert("RGB")
    text_prompt = prompt_builder(kind, row, image.width, image.height)
    user_content = [{"type": "image", "image": str(image_path)}, {"type": "text", "text": text_prompt}]
    if kind == "har":
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_content},
        ]
    else:
        messages = [{"role": "user", "content": user_content}]
    rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[rendered], images=[image], padding=True, return_tensors="pt").to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
    elapsed = time.perf_counter() - started
    trimmed = [
        output[len(input_ids) :] for input_ids, output in zip(inputs.input_ids, generated, strict=False)
    ]
    raw = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    return {
        "id": row["id"],
        "image_path": str(image_path),
        "image_sha256": sha256(image_path),
        "image_width": image.width,
        "image_height": image.height,
        "instruction": row.get("instruction"),
        "goal": row.get("goal"),
        "history": row.get("history"),
        "prompt": text_prompt,
        "raw_response": raw,
        "parsed_action": action_parser(raw, kind, image.width, image.height),
        "inference_s": elapsed,
        "input_tokens": int(inputs.input_ids.shape[-1]),
        "generated_tokens": int(generated.shape[-1] - inputs.input_ids.shape[-1]),
    }


def main() -> int:
    import torch
    import transformers

    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("uitars", "showui", "qwen25vl", "har"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest["rows"]
    if not rows or len({row["id"] for row in rows}) != len(rows):
        raise RuntimeError("manifest must contain non-empty rows with unique ids")
    model, processor = load_model(args.kind, args.model)
    outputs = []
    for index, row in enumerate(rows, 1):
        outputs.append(infer_one(model, processor, args.kind, row))
        print(f"processed {index}/{len(rows)}", flush=True)
    result = {
        "runner_version": RUNNER_VERSION,
        "kind": args.kind,
        "checkpoint": checkpoint_fingerprint(args.model),
        "manifest_path": str(args.manifest.resolve()),
        "manifest_sha256": sha256(args.manifest),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "generation": {
            "do_sample": False,
            "max_new_tokens": MAX_NEW_TOKENS,
            "torch_dtype": "bfloat16",
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
            "processor_use_fast": PROCESSOR_USE_FAST,
            "processor_padding_side": PROCESSOR_PADDING_SIDE,
        },
        "rows": outputs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.out)
    print(
        json.dumps(
            {
                "kind": args.kind,
                "rows": len(outputs),
                "valid_parses": sum((int(row["parsed_action"]["valid"]) for row in outputs)),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
