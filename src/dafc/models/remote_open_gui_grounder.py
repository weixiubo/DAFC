#!/usr/bin/env python3
"""Versioned offline batch runner for open GUI grounding checkpoints.

The runner is intentionally dataset-agnostic.  It reads a JSON manifest of
``id``, ``image_path``, and ``instruction`` records and writes raw generations
plus a deterministic normalized-coordinate parse.  DOM geometry, target
annotations, registration, authorization, and outcome scoring never enter the
remote process.
"""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import argparse
import ast
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
)


RUNNER_VERSION = "remote-open-gui-grounder-v2"
MIN_PIXELS = 128 * 28 * 28
MAX_PIXELS = 1008 * 28 * 28

UITARS_PROMPT = """You are a GUI action agent.
Given the screenshot and task, output the next executable action.

Action space:
- click(point='<point>x y</point>'): click the target at integer coordinates x y in a 0-1000 screen coordinate system.

Return one line starting with Action: and no extra explanation.
Task: {instruction}
"""

SHOWUI_PROMPT = """You are an assistant trained to navigate a GUI screen.
Given a task instruction and a screen observation, output the next action.
Return a dictionary exactly in this form:
{{'action': 'TAP', 'value': None, 'position': [x,y]}}
Position is relative to the full screenshot and must be scaled to 0-1.
Task: {instruction}
"""

QWEN_PROMPT = """You are a deterministic GUI grounding agent.
Find the requested target in the full screenshot and output exactly one click.
Return one line in this form and no explanation:
Action: click(point='[x,y]')
The full screenshot is {width} pixels wide and {height} pixels high. x and y
must be exact full-screenshot pixel coordinates: 0 <= x <= {width} and
0 <= y <= {height}. Do not normalize and do not return a bounding box.
Task: {instruction}
"""


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def checkpoint_fingerprint(path: Path) -> dict[str, Any]:
    config = path / "config.json"
    weights = sorted(
        item
        for pattern in ("*.safetensors", "*.bin")
        for item in path.glob(pattern)
        if item.is_file()
    )
    return {
        "path": str(path.resolve()),
        "config_sha256": sha256(config),
        "weight_files": [
            {"name": item.name, "bytes": item.stat().st_size} for item in weights
        ],
        "total_weight_bytes": sum(item.stat().st_size for item in weights),
    }


def prompt(kind: str, instruction: str, width: int, height: int) -> str:
    if kind == "uitars":
        return UITARS_PROMPT.format(instruction=instruction)
    if kind == "showui":
        return SHOWUI_PROMPT.format(instruction=instruction)
    if kind == "qwen25vl":
        return QWEN_PROMPT.format(
            instruction=instruction, width=width, height=height
        )
    raise ValueError(f"unknown model kind: {kind}")


def first_dictionary(text: str) -> dict[str, Any] | None:
    for candidate in re.findall(r"\{[^{}]*\}", text):
        try:
            parsed = ast.literal_eval(candidate)
        except Exception:
            try:
                parsed = json.loads(candidate)
            except Exception:
                continue
        if isinstance(parsed, dict):
            return parsed
    return None


def parse_normalized_click(
    text: str, kind: str, width: int, height: int
) -> dict[str, Any]:
    raw = text.strip()
    dictionary = first_dictionary(raw)
    if dictionary is not None:
        action = str(dictionary.get("action", "")).upper()
        position = dictionary.get("position")
        if action in {"TAP", "CLICK"} and isinstance(position, (list, tuple)) and len(position) >= 2:
            x, y = float(position[0]), float(position[1])
            if max(abs(x), abs(y)) > 1.0:
                if kind == "qwen25vl":
                    x, y = x / width, y / height
                else:
                    x, y = x / 1000.0, y / 1000.0
            valid = bool(math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1)
            return {
                "valid": valid,
                "parser": "action_dictionary",
                "normalized_xy": [x, y] if valid else None,
            }

    patterns = (
        r"<point>\s*([-+]?\d*\.?\d+)\s+([-+]?\d*\.?\d+)\s*</point>",
        r"<\|box_start\|>\s*\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)\s*<\|box_end\|>",
        r"click\s*\([^\d+-]*([-+]?\d*\.?\d+)\s*[, ]+\s*([-+]?\d*\.?\d+)",
        r"\[\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\]",
    )
    for pattern in patterns:
        match = re.search(pattern, raw, flags=re.IGNORECASE)
        if match is None:
            continue
        x, y = float(match.group(1)), float(match.group(2))
        if max(abs(x), abs(y)) > 1.0:
            if kind == "qwen25vl":
                x, y = x / width, y / height
            else:
                x, y = x / 1000.0, y / 1000.0
        valid = bool(math.isfinite(x) and math.isfinite(y) and 0 <= x <= 1 and 0 <= y <= 1)
        return {
            "valid": valid,
            "parser": "coordinate_regex",
            "normalized_xy": [x, y] if valid else None,
        }
    return {"valid": False, "parser": "parse_failure", "normalized_xy": None}


def load_model(kind: str, model_path: Path):
    processor = AutoProcessor.from_pretrained(
        model_path,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        local_files_only=True,
    )
    model_class = (
        Qwen2_5_VLForConditionalGeneration
        if kind == "qwen25vl"
        else Qwen2VLForConditionalGeneration
    )
    model = model_class.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=True,
    )
    model.eval()
    return model, processor


def infer_one(model, processor, kind: str, row: dict[str, Any]) -> dict[str, Any]:
    image_path = Path(row["image_path"])
    image = Image.open(image_path).convert("RGB")
    text_prompt = prompt(kind, str(row["instruction"]), image.width, image.height)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": text_prompt},
            ],
        }
    ]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=[rendered], images=[image], padding=True, return_tensors="pt"
    ).to(model.device)
    started = time.perf_counter()
    with torch.inference_mode():
        generated = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
        )
    elapsed = time.perf_counter() - started
    trimmed = [
        output[len(input_ids) :]
        for input_ids, output in zip(inputs.input_ids, generated)
    ]
    raw = processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]
    return {
        "id": row["id"],
        "image_path": str(image_path),
        "image_sha256": sha256(image_path),
        "image_width": image.width,
        "image_height": image.height,
        "instruction": row["instruction"],
        "prompt": text_prompt,
        "raw_response": raw,
        "parsed_action": parse_normalized_click(
            raw, kind, image.width, image.height
        ),
        "inference_s": elapsed,
        "input_tokens": int(inputs.input_ids.shape[-1]),
        "generated_tokens": int(generated.shape[-1] - inputs.input_ids.shape[-1]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=("uitars", "showui", "qwen25vl"), required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    rows = manifest["rows"]
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
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_name": torch.cuda.get_device_name(0),
        "generation": {
            "do_sample": False,
            "max_new_tokens": 128,
            "torch_dtype": "bfloat16",
            "min_pixels": MIN_PIXELS,
            "max_pixels": MAX_PIXELS,
        },
        "rows": outputs,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.out)
    print(
        json.dumps(
            {
                "kind": args.kind,
                "rows": len(outputs),
                "valid_parses": sum(
                    int(row["parsed_action"]["valid"]) for row in outputs
                ),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
