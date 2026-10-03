"""Local process boundary for the pinned LightGlue runtime front end.

AndroidWorld is pinned in a Python 3.11 environment, while the vision runtime
uses the project's Python 3.12 CUDA environment.  This module keeps those
dependency sets isolated: the CUDA environment serves correspondences on
127.0.0.1, and the AndroidWorld process uses ``LightGlueServiceClient`` as the
ordinary TransitionGuard match provider.

The wire format contains only PNG-encoded source/current screenshots and the
front end's correspondence receipt.  No task labels, rewards, accessibility
nodes, or target action annotations cross this boundary.
"""

from __future__ import annotations

import argparse
import base64
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib import error, request

import cv2
import numpy as np

SERVICE_VERSION = "transitionguard-lightglue-loopback-v1"
MAX_REQUEST_BYTES = 64 * 1024 * 1024


def _encode_png(image: np.ndarray) -> str:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("service image must be HxWx3 BGR")
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("could not PNG-encode service image")
    return base64.b64encode(encoded.tobytes()).decode("ascii")


def _decode_png(value: str) -> np.ndarray:
    try:
        encoded = np.frombuffer(base64.b64decode(value, validate=True), np.uint8)
    except (ValueError, TypeError) as exc:
        raise ValueError("invalid base64 PNG payload") from exc
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("could not decode HxWx3 PNG payload")
    return image


class LightGlueServiceClient:
    """Synchronous loopback client satisfying TransitionGuard's provider API."""

    def __init__(self, base_url: str, *, timeout_s: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_s = float(timeout_s)
        if not self.base_url.startswith("http://127.0.0.1:"):
            raise ValueError("LightGlue service must be bound to 127.0.0.1")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")

    def _json_request(self, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = None
        method = "GET"
        headers = {"Accept": "application/json"}
        if payload is not None:
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            method = "POST"
            headers["Content-Type"] = "application/json"
        req = request.Request(self.base_url + path, data=data, headers=headers, method=method)
        try:
            with request.urlopen(req, timeout=self.timeout_s) as response:
                body = response.read()
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LightGlue service HTTP {exc.code}: {detail}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"LightGlue service unavailable: {exc}") from exc
        decoded = json.loads(body)
        if not isinstance(decoded, dict):
            raise RuntimeError("LightGlue service returned a non-object receipt")
        return decoded

    def health(self) -> dict[str, Any]:
        return self._json_request("/health")

    def correspondences(self, source_bgr: np.ndarray, current_bgr: np.ndarray) -> dict[str, Any]:
        receipt = self._json_request(
            "/correspondences",
            {"source_png": _encode_png(source_bgr), "current_png": _encode_png(current_bgr)},
        )
        for key in ("source_points", "target_points"):
            points = np.asarray(receipt[key], dtype=np.float64)
            if points.size == 0:
                points = np.empty((0, 2), dtype=np.float64)
            if points.ndim != 2 or points.shape[1:] != (2,) or (not np.isfinite(points).all()):
                raise RuntimeError(f"LightGlue service returned invalid {key}")
            receipt[key] = points
        if receipt["source_points"].shape != receipt["target_points"].shape:
            raise RuntimeError("LightGlue point arrays have different shapes")
        matches = int(receipt["matches"])
        if matches != len(receipt["source_points"]):
            raise RuntimeError("LightGlue match count disagrees with points")
        scores = receipt.get("match_scores")
        receipt["match_scores"] = np.asarray(scores, dtype=np.float64) if scores is not None else None
        if receipt["match_scores"] is not None and (
            receipt["match_scores"].shape != (matches,) or not np.isfinite(receipt["match_scores"]).all()
        ):
            raise RuntimeError("LightGlue service returned invalid match scores")
        return receipt


def _jsonable_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    result = dict(receipt)
    for key in ("source_points", "target_points", "match_scores"):
        value = result.get(key)
        result[key] = value.tolist() if value is not None else None
    return result


def serve(host: str, port: int, *, repo: Path, weights: Path, device: str = "cuda") -> None:
    if host != "127.0.0.1":
        raise ValueError("refusing non-loopback LightGlue service binding")
    from dafc.models.lightglue_runtime_frontend import LightGlueRuntimeFrontend

    frontend = LightGlueRuntimeFrontend(repo=repo, weight_dir=weights, device=device)

    class Handler(BaseHTTPRequestHandler):
        server_version = SERVICE_VERSION

        def _send(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/health":
                self._send(404, {"error": "not_found"})
                return
            self._send(
                200, {"ready": True, "service_version": SERVICE_VERSION, "bindings": frontend.bindings}
            )

        def do_POST(self) -> None:
            if self.path != "/correspondences":
                self._send(404, {"error": "not_found"})
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if length <= 0 or length > MAX_REQUEST_BYTES:
                    raise ValueError("invalid request size")
                payload = json.loads(self.rfile.read(length))
                if set(payload) != {"source_png", "current_png"}:
                    raise ValueError("unexpected correspondence request fields")
                source = _decode_png(payload["source_png"])
                current = _decode_png(payload["current_png"])
                receipt = frontend.correspondences(source, current)
                self._send(200, _jsonable_receipt(receipt))
            except Exception as exc:
                self._send(400, {"error": type(exc).__name__, "detail": str(exc)})

        def log_message(self, format: str, *args: Any) -> None:
            print(f"{self.client_address[0]} {format % args}", flush=True)

    server = HTTPServer((host, port), Handler)
    print(
        json.dumps(
            {
                "ready": True,
                "service_version": SERVICE_VERSION,
                "host": host,
                "port": port,
                "bindings": frontend.bindings,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=48242)
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if not args.serve:
        parser.error("--serve is required")
    serve(args.host, args.port, repo=args.repo, weights=args.weights, device=args.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
