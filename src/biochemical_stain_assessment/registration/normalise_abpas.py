"""Macenko-style stain normalisation for AB-PAS brightfield images.

The built-in reference matrix was estimated from
``data_1_5/JH-311/AbPAS/ITG_Rusha_JH-311_Organoid_AbPAS_HMGU1-1.czi`` on
2026-05-13 using an 8x spatial subsample and the estimator implemented here.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_STAIN_MATRIX = np.array(
    [
        [0.7762909861832343, 0.5861256188305413, 0.23201091293552728],
        [0.7811455859120142, 0.5795937295446261, 0.23212643598420055],
    ],
    dtype=np.float64,
)
MAX_ESTIMATION_PIXELS = 1_000_000
NORMALISE_CHUNK_PIXELS = 1_000_000


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3).")
    return rgb[:, :, :3]


def _channel_imax(rgb: np.ndarray, percentile: float) -> np.ndarray:
    imax = np.percentile(rgb.astype(np.float64), percentile, axis=(0, 1))
    return np.maximum(imax, 1.0)


def _optical_density(rgb: np.ndarray, imax: np.ndarray) -> np.ndarray:
    rgb_f = np.minimum(rgb.astype(np.float64) + 1.0, imax)
    return -np.log(rgb_f / imax)


def _subsample_rgb(rgb: np.ndarray, max_pixels: int = MAX_ESTIMATION_PIXELS) -> np.ndarray:
    pixels = rgb.shape[0] * rgb.shape[1]
    if pixels <= max_pixels:
        return rgb
    stride = int(np.ceil(np.sqrt(pixels / max_pixels)))
    return rgb[::stride, ::stride]


def estimate_stain_matrix(
    rgb: np.ndarray,
    percentile: float = 99.0,
    angular_percentile: float = 99.0,
) -> np.ndarray:
    """
    Estimate the two dominant AB-PAS stain vectors from an RGB image.

    Parameters
    ----------
    rgb
        AB-PAS brightfield image with shape ``(H, W, 3)``.
    percentile
        Per-channel robust maximum used for optical-density conversion.
    angular_percentile
        Upper angular percentile. The lower percentile is ``100 - angular_percentile``.

    Returns
    -------
    ndarray
        ``(2, 3)`` float64 matrix with one unit OD stain vector per row.
    """
    rgb = _subsample_rgb(_validate_rgb(rgb))
    imax = _channel_imax(rgb, percentile)
    valid = np.all(rgb[:, :, :3] > 0, axis=-1)
    od = _optical_density(rgb, imax)
    tissue = valid & np.any(od > 0.15, axis=-1)
    od_flat = od[tissue]

    if od_flat.shape[0] < 3:
        raise ValueError("Cannot estimate stain matrix from a blank or near-blank image.")

    _, singular_values, vt = np.linalg.svd(od_flat, full_matrices=False)
    if singular_values.size < 2 or singular_values[1] <= 1e-8:
        raise ValueError("Cannot estimate two non-trivial stain vectors from this image.")

    plane = vt[:2]
    projected = od_flat @ plane.T
    angles = np.arctan2(projected[:, 1], projected[:, 0])
    lo_pct = max(0.0, 100.0 - angular_percentile)
    hi_pct = min(100.0, angular_percentile)
    lo_angle, hi_angle = np.percentile(angles, [lo_pct, hi_pct])

    v1 = plane.T @ np.array([np.cos(lo_angle), np.sin(lo_angle)])
    v2 = plane.T @ np.array([np.cos(hi_angle), np.sin(hi_angle)])
    matrix = np.vstack([v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)])

    # Stable row order keeps JSON output and tests deterministic.
    return matrix[np.argsort(matrix[:, 0])].astype(np.float64)


def normalise_abpas(
    rgb: np.ndarray,
    stain_matrix_target: np.ndarray | None = None,
    percentile: float = 99.0,
    angular_percentile: float = 99.0,
    clip: bool = True,
) -> np.ndarray:
    """
    Normalise an AB-PAS RGB image to a target Macenko stain matrix.

    Returns a uint8 RGB image with the same spatial shape as the input.
    """
    rgb = _validate_rgb(rgb)
    source = estimate_stain_matrix(rgb, percentile, angular_percentile)
    target = DEFAULT_STAIN_MATRIX if stain_matrix_target is None else np.asarray(
        stain_matrix_target,
        dtype=np.float64,
    )
    if target.shape != (2, 3):
        raise ValueError("stain_matrix_target must have shape (2, 3).")

    imax = _channel_imax(rgb, percentile)
    flat_rgb = rgb.reshape(-1, 3)
    out = np.empty((flat_rgb.shape[0], 3), dtype=np.float32)
    source_pinv = np.linalg.pinv(source)
    for start in range(0, flat_rgb.shape[0], NORMALISE_CHUNK_PIXELS):
        stop = min(start + NORMALISE_CHUNK_PIXELS, flat_rgb.shape[0])
        od = _optical_density(flat_rgb[start:stop], imax)
        concentrations = od @ source_pinv
        od_norm = concentrations @ target
        out[start:stop] = imax * np.exp(-od_norm)
    rgb_norm = out.reshape(rgb.shape)

    if clip:
        rgb_norm = np.clip(rgb_norm, 0, 255)
    return rgb_norm.astype(np.uint8)


def save_stain_matrix(
    matrix: np.ndarray,
    path: Path | str,
    sample_id: str,
    estimated_from: str,
) -> None:
    """Serialise a stain matrix and provenance to JSON."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.shape != (2, 3):
        raise ValueError("matrix must have shape (2, 3).")
    payload: dict[str, Any] = {
        "sample_id": sample_id,
        "estimated_from": estimated_from,
        "stain_matrix": matrix.tolist(),
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_stain_matrix(path: Path | str) -> np.ndarray:
    """Load a ``(2, 3)`` float64 stain matrix from a JSON sidecar."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    matrix = np.asarray(payload["stain_matrix"], dtype=np.float64)
    if matrix.shape != (2, 3):
        raise ValueError("Loaded stain matrix must have shape (2, 3).")
    return matrix
