"""Keypoint-based scale and translation registration."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from skimage import color, feature

from .affine_utils import apply_scale_translation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AffineParams:
    """Scale + translation affine transform with no rotation or shear."""

    scale: float
    tx: float
    ty: float
    n_inliers: int
    method: str
    rmse: float


def _as_uint8_gray(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Expected a 2-D grayscale image.")
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    else:
        arr = np.zeros_like(arr)
    return np.clip(arr * 255.0, 0, 255).astype(np.uint8)


def _extract_orb(img: np.ndarray, n_keypoints: int) -> tuple[np.ndarray, np.ndarray]:
    extractor = feature.ORB(n_keypoints=n_keypoints, fast_threshold=0.05)
    extractor.detect_and_extract(_as_uint8_gray(img))
    return extractor.keypoints.astype(np.float64), extractor.descriptors


def _extract_sift(img: np.ndarray, n_keypoints: int) -> tuple[np.ndarray, np.ndarray]:
    if not hasattr(feature, "SIFT"):
        raise RuntimeError("skimage.feature.SIFT is not available.")
    extractor = feature.SIFT(n_octaves=4, n_scales=3)
    extractor.detect_and_extract(_as_uint8_gray(img))
    keypoints = extractor.keypoints.astype(np.float64)
    descriptors = extractor.descriptors
    if len(keypoints) > n_keypoints:
        keypoints = keypoints[:n_keypoints]
        descriptors = descriptors[:n_keypoints]
    return keypoints, descriptors


def _match_with_detector(
    ref_gray: np.ndarray,
    mov_gray: np.ndarray,
    detector: str,
    n_keypoints: int,
    ratio_threshold: float,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        if detector.upper() == "ORB":
            ref_pts, ref_desc = _extract_orb(ref_gray, n_keypoints)
            mov_pts, mov_desc = _extract_orb(mov_gray, n_keypoints)
            metric = "hamming"
        elif detector.upper() == "SIFT":
            ref_pts, ref_desc = _extract_sift(ref_gray, n_keypoints)
            mov_pts, mov_desc = _extract_sift(mov_gray, n_keypoints)
            metric = "euclidean"
        else:
            raise ValueError("detector must be 'ORB' or 'SIFT'.")
    except (RuntimeError, ValueError):
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    if len(ref_pts) < 4 or len(mov_pts) < 4:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)

    matches = feature.match_descriptors(
        ref_desc,
        mov_desc,
        metric=metric,
        cross_check=True,
        max_ratio=ratio_threshold,
    )
    if len(matches) < 4:
        return np.empty((0, 2), dtype=np.float64), np.empty((0, 2), dtype=np.float64)
    return ref_pts[matches[:, 0]], mov_pts[matches[:, 1]]


def _detect_and_match_with_method(
    ref_gray: np.ndarray,
    mov_gray: np.ndarray,
    detector: str = "ORB",
    n_keypoints: int = 2000,
    ratio_threshold: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, str]:
    src_pts, dst_pts = _match_with_detector(
        ref_gray,
        mov_gray,
        detector,
        n_keypoints,
        ratio_threshold,
    )
    method = detector.upper()
    if detector.upper() == "ORB" and len(src_pts) < 4:
        src_pts, dst_pts = _match_with_detector(
            ref_gray,
            mov_gray,
            "SIFT",
            n_keypoints,
            ratio_threshold,
        )
        method = "SIFT" if len(src_pts) >= 4 else "ORB"
    return src_pts, dst_pts, method


def detect_and_match(
    ref_gray: np.ndarray,
    mov_gray: np.ndarray,
    detector: str = "ORB",
    n_keypoints: int = 2000,
    ratio_threshold: float = 0.75,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect keypoints and return matched ``(row, col)`` pairs."""
    src_pts, dst_pts, _ = _detect_and_match_with_method(
        ref_gray,
        mov_gray,
        detector,
        n_keypoints,
        ratio_threshold,
    )
    return src_pts, dst_pts


def _solve_scale_translation(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
) -> tuple[float, float, float]:
    n = len(src_pts)
    a = np.column_stack(
        [
            np.r_[dst_pts[:, 0], dst_pts[:, 1]],
            np.r_[np.ones(n), np.zeros(n)],
            np.r_[np.zeros(n), np.ones(n)],
        ]
    )
    b = np.r_[src_pts[:, 0], src_pts[:, 1]]
    params, _, _, _ = np.linalg.lstsq(a, b, rcond=None)
    scale, ty, tx = params
    return float(scale), float(tx), float(ty)


def _residuals(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    scale: float,
    tx: float,
    ty: float,
) -> np.ndarray:
    pred_row = scale * dst_pts[:, 0] + ty
    pred_col = scale * dst_pts[:, 1] + tx
    return np.sqrt((src_pts[:, 0] - pred_row) ** 2 + (src_pts[:, 1] - pred_col) ** 2)


def fit_scale_translation(
    src_pts: np.ndarray,
    dst_pts: np.ndarray,
    ransac_threshold: float = 3.0,
    min_inliers: int = 6,
) -> AffineParams | None:
    """Fit ``src ~= scale * dst + [ty, tx]`` with deterministic RANSAC."""
    src_pts = np.asarray(src_pts, dtype=np.float64)
    dst_pts = np.asarray(dst_pts, dtype=np.float64)
    if src_pts.shape != dst_pts.shape or src_pts.ndim != 2 or src_pts.shape[1] != 2:
        raise ValueError("src_pts and dst_pts must both have shape (N, 2).")
    if len(src_pts) < max(4, min_inliers):
        return None

    rng = np.random.default_rng(0)
    best_inliers: np.ndarray | None = None
    best_count = 0
    iterations = min(500, max(100, len(src_pts) * 4))

    for _ in range(iterations):
        sample = rng.choice(len(src_pts), size=4, replace=False)
        scale, tx, ty = _solve_scale_translation(src_pts[sample], dst_pts[sample])
        if not np.isfinite(scale) or scale <= 0:
            continue
        residual = _residuals(src_pts, dst_pts, scale, tx, ty)
        inliers = residual < ransac_threshold
        count = int(inliers.sum())
        if count > best_count:
            best_count = count
            best_inliers = inliers

    if best_inliers is None or best_count < min_inliers:
        return None

    scale, tx, ty = _solve_scale_translation(src_pts[best_inliers], dst_pts[best_inliers])
    residual = _residuals(src_pts[best_inliers], dst_pts[best_inliers], scale, tx, ty)
    rmse = float(np.sqrt(np.mean(residual**2))) if residual.size else float("inf")
    return AffineParams(
        scale=float(scale),
        tx=float(tx),
        ty=float(ty),
        n_inliers=best_count,
        method="keypoint",
        rmse=rmse,
    )


def _abpas_to_inverted_luminance(abpas_rgb: np.ndarray) -> np.ndarray:
    rgb = np.asarray(abpas_rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("abpas_rgb must have shape (H, W, 3).")
    gray = color.rgb2gray(rgb[:, :, :3].astype(np.float32) / 255.0)
    return (1.0 - gray).astype(np.float32)


def register_keypoint(
    abpas_rgb: np.ndarray,
    dapi: np.ndarray,
    detector: str = "ORB",
    n_keypoints: int = 2000,
    ratio_threshold: float = 0.75,
    ransac_threshold: float = 3.0,
    min_inliers: int = 6,
    fallback_params: AffineParams | None = None,
) -> tuple[np.ndarray, np.ndarray, AffineParams]:
    """Register DAPI to AB-PAS using keypoints and a scale-translation transform."""
    ref_gray = _abpas_to_inverted_luminance(abpas_rgb)
    mov_gray = np.asarray(dapi, dtype=np.float32)
    src_pts, dst_pts, method = _detect_and_match_with_method(
        ref_gray,
        mov_gray,
        detector=detector,
        n_keypoints=n_keypoints,
        ratio_threshold=ratio_threshold,
    )
    params = fit_scale_translation(
        src_pts,
        dst_pts,
        ransac_threshold=ransac_threshold,
        min_inliers=min_inliers,
    )
    if params is None:
        params = fallback_params or AffineParams(
            scale=1.0,
            tx=0.0,
            ty=0.0,
            n_inliers=0,
            method="stage_coords",
            rmse=0.0,
        )
        logger.warning("Keypoint registration failed; using %s fallback.", params.method)
    else:
        params = AffineParams(
            scale=params.scale,
            tx=params.tx,
            ty=params.ty,
            n_inliers=params.n_inliers,
            method=method,
            rmse=params.rmse,
        )

    dapi_aligned = apply_scale_translation(
        mov_gray,
        params.scale,
        params.tx,
        params.ty,
        output_shape=abpas_rgb.shape[:2],
        order=1,
    )
    return abpas_rgb, dapi_aligned, params
