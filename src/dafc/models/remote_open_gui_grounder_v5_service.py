#!/usr/bin/env python3
"""Persistent loopback HTTP service for closed-loop GUI grounding.

The frozen V4 batch runner reloads a checkpoint for every manifest, which is
unsuitable for an agent that must alternate between a live environment and a
remote GPU.  V5 keeps V3/V4 prompts, parsers, processor settings, and generation
unchanged, but loads one checkpoint once and accepts one screenshot request at
a time over a loopback-only HTTP endpoint.

The service is intended to be reached through an SSH local port forward.  It
refuses non-loopback binding, verifies every transmitted image digest, writes a
durable JSONL audit before returning a response, and rejects duplicate request
IDs.  Target DOM, task rewards, memory decisions, and scoring annotations never
enter this process.
"""

from __future__ import annotations
from importlib import import_module
from dafc.optional import LazyModule

import argparse
import base64
import hashlib
import json
import os
import tempfile
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import dafc.models.remote_open_gui_grounder_v3 as v3


SERVICE_VERSION = "remote-open-gui-grounder-v5-service"
BASE_RUNNER_VERSION = v3.RUNNER_VERSION
LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}
MAX_REQUEST_BYTES = 24 * 1024 * 1024


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def bytes_sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    result = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise RuntimeError(
                    f"invalid audit JSONL at line {line_number}: {error}"
                ) from error
            request_id = str(row.get("request", {}).get("id", ""))
            if not request_id or request_id in result:
                raise RuntimeError(
                    f"missing or duplicate request id in audit: {request_id!r}"
                )
            result.add(request_id)
    return result


def decode_request(payload: dict[str, Any], kind: str) -> tuple[dict[str, Any], bytes, str]:
    request_id = str(payload.get("id", "")).strip()
    encoded = payload.get("image_base64")
    claimed_sha256 = str(payload.get("image_sha256", "")).lower().strip()
    image_format = str(payload.get("image_format", "png")).lower().strip()
    if not request_id:
        raise ValueError("request requires a non-empty id")
    if not isinstance(encoded, str) or not encoded:
        raise ValueError("request requires image_base64")
    if image_format not in {"png", "jpeg", "jpg"}:
        raise ValueError(f"unsupported image_format: {image_format}")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except Exception as error:
        raise ValueError("image_base64 is invalid") from error
    observed_sha256 = bytes_sha256(image_bytes)
    if len(claimed_sha256) != 64 or claimed_sha256 != observed_sha256:
        raise ValueError("image_sha256 does not match transmitted bytes")

    row: dict[str, Any] = {"id": request_id}
    if kind == "har":
        goal = str(payload.get("goal", "")).strip()
        history = str(payload.get("history", "")).strip()
        if not goal or not history:
            raise ValueError("HAR requests require non-empty goal and history")
        row.update({"goal": goal, "history": history})
    else:
        instruction = str(payload.get("instruction", "")).strip()
        if not instruction:
            raise ValueError("non-HAR requests require a non-empty instruction")
        row["instruction"] = instruction
    return row, image_bytes, "jpg" if image_format == "jpg" else image_format


def durable_append(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = canonical_json(row) + b"\n"
    with path.open("ab", buffering=0) as handle:
        handle.write(payload)
        os.fsync(handle.fileno())


class GrounderServer(HTTPServer):
    """Single-threaded server; GPU requests cannot overlap."""

    def __init__(
        self,
        address: tuple[str, int],
        model,
        processor,
        kind: str,
        audit_path: Path,
        checkpoint: dict[str, Any],
    ):
        super().__init__(address, GrounderHandler)
        self.model = model
        self.processor = processor
        self.kind = kind
        self.audit_path = audit_path
        self.checkpoint = checkpoint
        self.completed_ids = existing_ids(audit_path)
        self.started_unix_s = time.time()


class GrounderHandler(BaseHTTPRequestHandler):
    server: GrounderServer

    def log_message(self, format: str, *args: Any) -> None:
        print(
            f"http client={self.client_address[0]} "
            f"{format % args}",
            flush=True,
        )

    def send_json(self, status: HTTPStatus, body: dict[str, Any]) -> None:
        payload = canonical_json(body)
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path != "/health":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        self.send_json(
            HTTPStatus.OK,
            {
                "ready": True,
                "service_version": SERVICE_VERSION,
                "base_runner_version": BASE_RUNNER_VERSION,
                "kind": self.server.kind,
                "completed_requests": len(self.server.completed_ids),
                "started_unix_s": self.server.started_unix_s,
                "checkpoint": self.server.checkpoint,
            },
        )

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        if self.path != "/infer":
            self.send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            content_length = 0
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            self.send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": "invalid_content_length"},
            )
            return
        received = self.rfile.read(content_length)
        request_sha256 = bytes_sha256(received)
        try:
            payload = json.loads(received)
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            row, image_bytes, extension = decode_request(
                payload, self.server.kind
            )
            if row["id"] in self.server.completed_ids:
                raise FileExistsError(f"duplicate request id: {row['id']}")
        except FileExistsError as error:
            self.send_json(
                HTTPStatus.CONFLICT,
                {"error": "duplicate_request_id", "detail": str(error)},
            )
            return
        except Exception as error:
            self.send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": "invalid_request", "detail": str(error)},
            )
            return

        started = time.perf_counter()
        try:
            with tempfile.TemporaryDirectory(prefix="gui-grounder-v5-") as root:
                image_path = Path(root) / f"observation.{extension}"
                image_path.write_bytes(image_bytes)
                inference_row = {**row, "image_path": str(image_path)}
                output = v3.infer_one(
                    self.server.model,
                    self.server.processor,
                    self.server.kind,
                    inference_row,
                )
            service_s = time.perf_counter() - started
            audit_row = {
                "service_version": SERVICE_VERSION,
                "base_runner_version": BASE_RUNNER_VERSION,
                "kind": self.server.kind,
                "request": {
                    "id": row["id"],
                    "request_sha256": request_sha256,
                    "image_sha256": output["image_sha256"],
                    "instruction": row.get("instruction"),
                    "goal": row.get("goal"),
                    "history": row.get("history"),
                },
                "response": output,
                "service_s": service_s,
                "completed_unix_s": time.time(),
            }
            durable_append(self.server.audit_path, audit_row)
            self.server.completed_ids.add(row["id"])
            self.send_json(
                HTTPStatus.OK,
                {
                    "service_version": SERVICE_VERSION,
                    "kind": self.server.kind,
                    "request_sha256": request_sha256,
                    "service_s": service_s,
                    "result": output,
                },
            )
        except Exception as error:
            self.send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "inference_failure", "detail": repr(error)},
            )


def main() -> int:
    import torch
    import transformers

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--kind", choices=("uitars", "showui", "qwen25vl", "har"), required=True
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    if args.host not in LOOPBACK_HOSTS:
        raise ValueError("V5 service may bind only to a loopback host")
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be between 1 and 65535")

    v3.torch = torch
    model, processor = v3.load_model(args.kind, args.model)
    checkpoint = v3.checkpoint_fingerprint(args.model)
    server = GrounderServer(
        (args.host, args.port),
        model,
        processor,
        args.kind,
        args.audit,
        checkpoint,
    )
    print(
        json.dumps(
            {
                "ready": True,
                "service_version": SERVICE_VERSION,
                "base_runner_version": BASE_RUNNER_VERSION,
                "kind": args.kind,
                "host": args.host,
                "port": args.port,
                "audit": str(args.audit.resolve()),
                "checkpoint": checkpoint,
                "torch_version": torch.__version__,
                "transformers_version": transformers.__version__,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cuda_device_name": torch.cuda.get_device_name(0),
                "generation": {
                    "do_sample": False,
                    "max_new_tokens": v3.MAX_NEW_TOKENS,
                    "torch_dtype": "bfloat16",
                    "min_pixels": v3.MIN_PIXELS,
                    "max_pixels": v3.MAX_PIXELS,
                    "processor_use_fast": v3.PROCESSOR_USE_FAST,
                    "processor_padding_side": v3.PROCESSOR_PADDING_SIDE,
                },
            },
            sort_keys=True,
        ),
        flush=True,
    )
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
