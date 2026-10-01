#!/usr/bin/env python3
"""Hash-verifying client for remote_open_gui_grounder_v5_service.py."""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import argparse
import base64
import hashlib
import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def infer(
    base_url: str,
    request_id: str,
    image_path: Path,
    *,
    instruction: str | None = None,
    goal: str | None = None,
    history: str | None = None,
    timeout_s: float = 180.0,
) -> dict[str, Any]:
    image_bytes = image_path.read_bytes()
    suffix = image_path.suffix.lower()
    image_format = "jpeg" if suffix in {".jpg", ".jpeg"} else "png"
    payload: dict[str, Any] = {
        "id": request_id,
        "image_base64": base64.b64encode(image_bytes).decode("ascii"),
        "image_sha256": sha256_bytes(image_bytes),
        "image_format": image_format,
    }
    if instruction is not None:
        payload["instruction"] = instruction
    if goal is not None:
        payload["goal"] = goal
    if history is not None:
        payload["history"] = history
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/infer",
        data=encoded,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            result = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"grounder HTTP {error.code}: {detail}") from error
    observed = result.get("result", {}).get("image_sha256")
    if observed != payload["image_sha256"]:
        raise RuntimeError(
            f"service image hash mismatch: {observed} != {payload['image_sha256']}"
        )
    if result.get("result", {}).get("id") != request_id:
        raise RuntimeError("service response id mismatch")
    return result


def health(base_url: str, timeout_s: float = 10.0) -> dict[str, Any]:
    with urllib.request.urlopen(
        f"{base_url.rstrip('/')}/health", timeout=timeout_s
    ) as response:
        return json.loads(response.read())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True)
    parser.add_argument("--health", action="store_true")
    parser.add_argument("--id")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--instruction")
    parser.add_argument("--goal")
    parser.add_argument("--history")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--timeout-s", type=float, default=180.0)
    args = parser.parse_args()
    if args.health:
        result = health(args.url, args.timeout_s)
    else:
        if not args.id or args.image is None:
            raise ValueError("inference requires --id and --image")
        result = infer(
            args.url,
            args.id,
            args.image,
            instruction=args.instruction,
            goal=args.goal,
            history=args.history,
            timeout_s=args.timeout_s,
        )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.out is None:
        print(rendered, end="")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.out.with_suffix(args.out.suffix + ".tmp")
        temporary.write_text(rendered, encoding="utf-8")
        temporary.replace(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
