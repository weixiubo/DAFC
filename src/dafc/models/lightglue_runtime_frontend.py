"""Pinned SuperPoint+LightGlue correspondences for online TransitionGuard.

The implementation follows the already preregistered C424 front end: official
LightGlue commit eb42fee, SuperPoint(max_num_keypoints=2048), official matcher
defaults, no confidence post-filtering, deterministic inference, and locally
hash-bound weights.  Source features are cached across an episode and the most
recent query features are reused while TransitionGuard evaluates several
candidate transitions against one current screenshot.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_REPO_COMMIT = "eb42fee2d71449efb0aa5c10549752b5d75384d8"
EXPECTED_WEIGHT_HASHES = {
    "superpoint_v1.pth": "52b6708629640ca883673b5d5c097c4ddad37d8048b33f09c8ca0d69db12c40e",
    "superpoint_lightglue.pth": "6ff7040d0a497fc6639337946d7538dae07428c18f77a067a0b5a960e7cc551a",
}
FRONTEND_VERSION = "pinned-superpoint-lightglue-runtime-v1"
MAX_KEYPOINTS = 2048
SOURCE_CACHE_LIMIT = 128
PAIR_CACHE_LIMIT = 4096


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(image: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(str(image.dtype).encode("ascii"))
    digest.update(np.asarray(image.shape, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(image).tobytes())
    return digest.hexdigest()


def repo_commit(path: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"], check=True, capture_output=True, text=True
    ).stdout.strip()


class LightGlueRuntimeFrontend:
    """Online learned correspondence provider with auditable bindings."""

    def __init__(
        self,
        *,
        repo: Path,
        weight_dir: Path,
        device: str = "cuda",
        source_cache_limit: int = SOURCE_CACHE_LIMIT,
        pair_cache_limit: int = PAIR_CACHE_LIMIT,
    ) -> None:
        self.repo = repo.resolve()
        self.superpoint_weights = (weight_dir / "superpoint_v1.pth").resolve()
        self.lightglue_weights = (weight_dir / "superpoint_lightglue.pth").resolve()
        self.source_cache_limit = int(source_cache_limit)
        self.pair_cache_limit = int(pair_cache_limit)
        if self.source_cache_limit <= 0:
            raise ValueError("source_cache_limit must be positive")
        if self.pair_cache_limit <= 0:
            raise ValueError("pair_cache_limit must be positive")
        self._verify_bindings()
        if str(self.repo) not in sys.path:
            sys.path.insert(0, str(self.repo))
        import kornia
        import torch

        if device == "cuda" and (not torch.cuda.is_available()):
            raise RuntimeError("LightGlue runtime requested CUDA but CUDA is unavailable")
        self.torch = torch
        self.device = torch.device(device)
        torch.manual_seed(424001)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(424001)
        torch.backends.cudnn.benchmark = False
        original_loader = torch.hub.load_state_dict_from_url

        def local_loader(url: str, *args, **kwargs):
            if "superpoint_v1.pth" in url:
                path = self.superpoint_weights
            elif "superpoint_lightglue.pth" in url:
                path = self.lightglue_weights
            else:
                raise RuntimeError(f"unbound LightGlue model URL: {url}")
            return torch.load(path, map_location="cpu", weights_only=True)

        torch.hub.load_state_dict_from_url = local_loader
        try:
            from lightglue import LightGlue, SuperPoint
            from lightglue.utils import numpy_image_to_torch, rbd

            self.extractor = SuperPoint(max_num_keypoints=MAX_KEYPOINTS).eval().to(self.device)
            self.matcher = LightGlue(features="superpoint").eval().to(self.device)
            self.numpy_image_to_torch = numpy_image_to_torch
            self.rbd = rbd
        finally:
            torch.hub.load_state_dict_from_url = original_loader
        self.environment = {
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "kornia": kornia.__version__,
            "device": str(self.device),
            "cuda_device_name": torch.cuda.get_device_name(self.device)
            if self.device.type == "cuda"
            else None,
            "rng_seed": 424001,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
        }
        self._source_cache: OrderedDict[str, dict[str, Any]] = OrderedDict()
        self._pair_cache: OrderedDict[tuple[str, str], dict[str, Any]] = OrderedDict()
        self._query_key: str | None = None
        self._query_features: dict[str, Any] | None = None

    def _verify_bindings(self) -> None:
        if not self.repo.is_dir():
            raise RuntimeError(f"LightGlue repository is missing: {self.repo}")
        observed_commit = repo_commit(self.repo)
        if observed_commit != EXPECTED_REPO_COMMIT:
            raise RuntimeError(f"LightGlue commit mismatch: {observed_commit} != {EXPECTED_REPO_COMMIT}")
        for path in (self.superpoint_weights, self.lightglue_weights):
            if not path.is_file():
                raise RuntimeError(f"LightGlue weight is missing: {path}")
            observed = sha256(path)
            expected = EXPECTED_WEIGHT_HASHES[path.name]
            if observed != expected:
                raise RuntimeError(
                    f"LightGlue weight hash mismatch for {path.name}: {observed} != {expected}"
                )

    @property
    def bindings(self) -> dict[str, Any]:
        return {
            "frontend_version": FRONTEND_VERSION,
            "repo": {"path": str(self.repo), "commit": repo_commit(self.repo)},
            "weights": {
                path.name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                for path in (self.superpoint_weights, self.lightglue_weights)
            },
            "official_defaults": {
                "superpoint_max_num_keypoints": MAX_KEYPOINTS,
                "superpoint_resize": 1024,
                "lightglue_n_layers": 9,
                "lightglue_depth_confidence": 0.95,
                "lightglue_width_confidence": 0.99,
                "lightglue_filter_threshold": 0.1,
                "mixed_precision": False,
                "additional_confidence_filter": None,
            },
            "environment": self.environment,
        }

    def _tensor(self, image_bgr: np.ndarray):
        if image_bgr.ndim != 3 or image_bgr.shape[2] != 3:
            raise ValueError("LightGlue input must be an HxWx3 BGR image")
        rgb = np.ascontiguousarray(image_bgr[..., ::-1])
        return self.numpy_image_to_torch(rgb).to(self.device)

    def _extract(self, image_bgr: np.ndarray) -> dict[str, Any]:
        return self.extractor.extract(self._tensor(image_bgr))

    def _source_features(self, image_bgr: np.ndarray) -> tuple[str, dict[str, Any]]:
        key = array_sha256(image_bgr)
        cached = self._source_cache.get(key)
        if cached is not None:
            self._source_cache.move_to_end(key)
            return (key, cached)
        features = self._extract(image_bgr)
        self._source_cache[key] = features
        self._source_cache.move_to_end(key)
        while len(self._source_cache) > self.source_cache_limit:
            self._source_cache.popitem(last=False)
        return (key, features)

    def _current_features(self, image_bgr: np.ndarray) -> tuple[str, dict[str, Any]]:
        key = array_sha256(image_bgr)
        if key != self._query_key or self._query_features is None:
            self._query_features = self._extract(image_bgr)
            self._query_key = key
        return (key, self._query_features)

    def correspondences(self, source_bgr: np.ndarray, current_bgr: np.ndarray) -> dict[str, Any]:
        """Return all official matcher correspondences without post-filtering."""
        started = time.perf_counter()
        source_array_key = array_sha256(source_bgr)
        current_array_key = array_sha256(current_bgr)
        pair_key = (source_array_key, current_array_key)
        cached = self._pair_cache.get(pair_key)
        if cached is not None:
            self._pair_cache.move_to_end(pair_key)
            return {**cached, "cache_hit": True, "cache_lookup_s": time.perf_counter() - started}
        with self.torch.inference_mode():
            source_key, source_features = self._source_features(source_bgr)
            current_key, current_features = self._current_features(current_bgr)
            matched = self.matcher({"image0": source_features, "image1": current_features})
            if self.device.type == "cuda":
                self.torch.cuda.synchronize(self.device)
            source_unbatched, current_unbatched, match_unbatched = [
                self.rbd(row) for row in (source_features, current_features, matched)
            ]
            indices = match_unbatched["matches"]
            source_points = source_unbatched["keypoints"][indices[..., 0]]
            current_points = current_unbatched["keypoints"][indices[..., 1]]
            if not bool(self.torch.isfinite(source_points).all()) or not bool(
                self.torch.isfinite(current_points).all()
            ):
                raise RuntimeError("LightGlue returned non-finite correspondences")
            scores = match_unbatched.get("scores")
            receipt = {
                "frontend_version": FRONTEND_VERSION,
                "source_image_array_sha256": source_key,
                "current_image_array_sha256": current_key,
                "source_keypoints": int(len(source_unbatched["keypoints"])),
                "target_keypoints": int(len(current_unbatched["keypoints"])),
                "matches": int(len(indices)),
                "source_points": source_points.detach().cpu().numpy(),
                "target_points": current_points.detach().cpu().numpy(),
                "match_scores": scores.detach().cpu().numpy() if scores is not None else None,
                "inference_s": time.perf_counter() - started,
                "cache_hit": False,
                "cache_lookup_s": None,
            }
            self._pair_cache[pair_key] = receipt
            self._pair_cache.move_to_end(pair_key)
            while len(self._pair_cache) > self.pair_cache_limit:
                self._pair_cache.popitem(last=False)
            del matched
            return receipt

    def compact_receipt(self, receipt: dict[str, Any]) -> dict[str, Any]:
        scores = receipt.get("match_scores")
        return {
            "frontend_version": receipt["frontend_version"],
            "source_image_array_sha256": receipt["source_image_array_sha256"],
            "current_image_array_sha256": receipt["current_image_array_sha256"],
            "source_keypoints": receipt["source_keypoints"],
            "target_keypoints": receipt["target_keypoints"],
            "matches": receipt["matches"],
            "match_score_min": float(np.min(scores)) if scores is not None and len(scores) else None,
            "match_score_median": float(np.median(scores)) if scores is not None and len(scores) else None,
            "match_score_max": float(np.max(scores)) if scores is not None and len(scores) else None,
            "inference_s": receipt["inference_s"],
            "cache_hit": receipt.get("cache_hit", False),
            "cache_lookup_s": receipt.get("cache_lookup_s"),
        }

    def print_bindings(self) -> None:
        print(json.dumps(self.bindings, indent=2, sort_keys=True))


def main() -> int:
    frontend = LightGlueRuntimeFrontend()
    frontend.print_bindings()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
