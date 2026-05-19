"""Contrast enhancement for DAPI fluorescence images."""

from __future__ import annotations

import logging

import numpy as np
from skimage import exposure, morphology

logger = logging.getLogger(__name__)


def clip_percentile(img: np.ndarray, low: float = 0.1, high: float = 99.9) -> np.ndarray:
    """Clip an image to percentile bounds and rescale it to float32 ``[0, 1]``."""
    arr = np.asarray(img, dtype=np.float32)
    if arr.size == 0:
        return arr.astype(np.float32)

    lo, hi = np.percentile(arr, [low, high])
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.float32)

    clipped = np.clip(arr, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def enhance_dapi(
    dapi: np.ndarray,
    clip_low: float = 0.1,
    clip_high: float = 99.9,
    clahe_kernel_size: int | None = None,
    clahe_clip_limit: float = 0.01,
    tophat: bool = False,
    tophat_radius: int = 15,
) -> np.ndarray:
    """Enhance DAPI contrast with percentile clipping, CLAHE, and optional top-hat."""
    base = clip_percentile(dapi, clip_low, clip_high)
    if not np.any(base):
        return base

    kernel_size = clahe_kernel_size
    if kernel_size is None:
        kernel_size = max(1, min(base.shape) // 8)
    logger.debug("DAPI CLAHE kernel size: %s", kernel_size)

    enhanced = exposure.equalize_adapthist(
        base,
        kernel_size=kernel_size,
        clip_limit=clahe_clip_limit,
    ).astype(np.float32)

    if tophat:
        footprint = morphology.disk(tophat_radius)
        enhanced = morphology.white_tophat(enhanced, footprint=footprint).astype(np.float32)
        if enhanced.max() > enhanced.min():
            enhanced = (enhanced - enhanced.min()) / (enhanced.max() - enhanced.min())

    return np.clip(enhanced, 0.0, 1.0).astype(np.float32)
