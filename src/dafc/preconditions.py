from __future__ import annotations

from typing import Any

import cv2
import numpy as np

MIN_NORMALIZED_MARGIN = 0.2
MAX_CONTEXT_ERROR = 0.12


def apply_geometry(image: np.ndarray, geometry: str) -> np.ndarray:
    height, width = image.shape[:2]
    if geometry == "identity":
        return image.copy()
    if geometry == "zoom_125":
        return cv2.resize(image, (round(width * 1.25), round(height * 1.25)), interpolation=cv2.INTER_LINEAR)
    if geometry == "zoom_80":
        return cv2.resize(image, (round(width * 0.8), round(height * 0.8)), interpolation=cv2.INTER_AREA)
    if geometry == "affine":
        matrix = np.asarray([[1.02, -0.05, 54.0], [0.08, 0.96, 24.0]], dtype=np.float32)
        return cv2.warpAffine(
            image,
            matrix,
            (width + 54, height + 48),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(255, 255, 255),
        )
    raise KeyError(geometry)


def jpeg_roundtrip(image: np.ndarray, quality: int | None) -> np.ndarray:
    if quality is None:
        return image
    ok, payload = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    decoded = cv2.imdecode(payload, cv2.IMREAD_COLOR)
    if decoded is None:
        raise RuntimeError("JPEG decoding failed")
    return decoded


def render_template(image: np.ndarray, variant: dict[str, Any]) -> np.ndarray:
    """Warp a stored state with the geometry map accepted by gate G."""
    return apply_geometry(image, str(variant["geometry"]))


def render_query(image: np.ndarray, variant: dict[str, Any]) -> np.ndarray:
    """Render a target observation, including target-stream compression."""
    warped = apply_geometry(image, str(variant["geometry"]))
    return jpeg_roundtrip(warped, variant["jpeg_quality"])


def effect_model(before: np.ndarray, after: np.ndarray) -> dict[str, Any]:
    before_f = before.astype(np.float32)
    after_f = after.astype(np.float32)
    channel_delta = np.abs(after_f - before_f)
    delta = np.max(channel_delta, axis=2)
    core = delta > PIXEL_EFFECT_THRESHOLD
    core_pixels = int(core.sum())
    weights = np.where(core, delta, 0.0).astype(np.float32)
    weight_sum = float(weights.sum())
    source_pixel_error = np.mean(channel_delta, axis=2)
    separation = float((weights * source_pixel_error).sum() / weight_sum) if weight_sum > 0.0 else 0.0
    core_u8 = core.astype(np.uint8)
    outer = cv2.dilate(core_u8, np.ones((13, 13), np.uint8), iterations=1) > 0
    inner = cv2.dilate(core_u8, np.ones((3, 3), np.uint8), iterations=1) > 0
    context = outer & ~inner
    observable = bool(core_pixels >= MIN_EFFECT_PIXELS and separation >= MIN_EFFECT_SEPARATION)
    return {
        "weights": weights,
        "context": context,
        "core_pixels": core_pixels,
        "source_separation": separation,
        "observable": observable,
    }


def score_query(
    source_before: np.ndarray,
    source_after: np.ndarray,
    query: np.ndarray,
    model: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if source_before.shape != source_after.shape or query.shape != source_before.shape:
        raise RuntimeError(f"shape mismatch: {source_before.shape} {source_after.shape} {query.shape}")
    model = model or effect_model(source_before, source_after)
    if not model["observable"]:
        return {
            "observable": False,
            "accepted": False,
            "reason": "unobservable_source_effect",
            "core_pixels": model["core_pixels"],
            "source_separation": model["source_separation"],
            "pre_error": None,
            "post_error": None,
            "normalized_margin": None,
            "context_error": None,
        }
    before_f = source_before.astype(np.float32)
    after_f = source_after.astype(np.float32)
    query_f = query.astype(np.float32)
    weights = model["weights"]
    weight_sum = float(weights.sum())
    pre_pixel_error = np.mean(np.abs(query_f - before_f), axis=2)
    post_pixel_error = np.mean(np.abs(query_f - after_f), axis=2)
    pre_error = float((weights * pre_pixel_error).sum() / weight_sum)
    post_error = float((weights * post_pixel_error).sum() / weight_sum)
    normalized_margin = float((post_error - pre_error) / max(float(model["source_separation"]), 1e-06))
    context = model["context"]
    if int(context.sum()) >= 16:
        context_error = float(np.mean(np.abs(query_f[context] - before_f[context])) / 255.0)
    else:
        context_error = 0.0
    margin_ok = normalized_margin >= MIN_NORMALIZED_MARGIN
    context_ok = context_error <= MAX_CONTEXT_ERROR
    accepted = bool(margin_ok and context_ok)
    if margin_ok and (not context_ok):
        reason = "context_mismatch"
    elif margin_ok:
        reason = "applicable_pre_state"
    elif normalized_margin <= -MIN_NORMALIZED_MARGIN and context_ok:
        reason = "confident_post_state"
    elif normalized_margin <= -MIN_NORMALIZED_MARGIN:
        reason = "post_state_context_mismatch"
    else:
        reason = "ambiguous_state"
    return {
        "observable": True,
        "accepted": accepted,
        "reason": reason,
        "core_pixels": model["core_pixels"],
        "source_separation": model["source_separation"],
        "pre_error": pre_error,
        "post_error": post_error,
        "normalized_margin": normalized_margin,
        "context_error": context_error,
    }


PIXEL_EFFECT_THRESHOLD = 12.0
MIN_EFFECT_PIXELS = 16
MIN_EFFECT_SEPARATION = 8.0
VARIANTS: tuple[dict[str, Any], ...] = (
    {"name": "identity", "geometry": "identity", "jpeg_quality": None},
    {"name": "jpeg_82", "geometry": "identity", "jpeg_quality": 82},
    {"name": "zoom_125_jpeg_82", "geometry": "zoom_125", "jpeg_quality": 82},
    {"name": "zoom_80_jpeg_82", "geometry": "zoom_80", "jpeg_quality": 82},
    {"name": "affine_jpeg_82", "geometry": "affine", "jpeg_quality": 82},
)
