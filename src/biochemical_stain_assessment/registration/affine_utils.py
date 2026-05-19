"""Helpers for scale-translation affine transforms and sidecar I/O."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .keypoint_register import AffineParams


def apply_scale_translation(
    img: np.ndarray,
    scale: float,
    tx: float,
    ty: float,
    output_shape: tuple[int, int],
    order: int = 1,
) -> np.ndarray:
    """Warp an image onto the target grid using isotropic scale and translation."""
    try:
        import cv2
        
        M = np.array([
            [scale, 0, tx],
            [0, scale, ty]
        ], dtype=np.float32)

        warped = cv2.warpAffine(img, M, (output_shape[1], output_shape[0]))

    except ImportError:
        from skimage.transform import AffineTransform, warp

        transform = AffineTransform(scale=(scale, scale), translation=(tx, ty))
        warped = warp(
            img,
            inverse_map=transform.inverse,
            output_shape=output_shape,
            order=order,
            mode="reflect",
            preserve_range=True,
        )
        if np.issubdtype(np.asarray(img).dtype, np.floating):
            return np.clip(warped, 0.0, 1.0).astype(np.float32)
    return warped.astype(img.dtype, copy=False)


def mutual_crop(
    abpas: np.ndarray,
    dapi_warped: np.ndarray,
    margin: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim the same edge margin from both registered images."""
    if abpas.shape[:2] != dapi_warped.shape[:2]:
        raise ValueError("abpas and dapi_warped must have the same spatial shape.")
    if margin < 0:
        raise ValueError("margin must be non-negative.")
    h, w = abpas.shape[:2]
    if margin == 0:
        return abpas, dapi_warped
    if 2 * margin >= h or 2 * margin >= w:
        raise ValueError("margin is too large for the image dimensions.")
    return abpas[margin:-margin, margin:-margin], dapi_warped[margin:-margin, margin:-margin]


def save_affine_params(
    params: "AffineParams",
    path: Path | str,
    sample_id: str,
    abpas_path: str,
    dapi_path: str,
) -> None:
    """Serialise affine parameters and provenance to JSON."""
    payload: dict[str, Any] = {
        "sample_id": sample_id,
        "abpas_path": str(abpas_path),
        "dapi_path": str(dapi_path),
        "affine_params": asdict(params),
    }
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_affine_params(path: Path | str) -> "AffineParams":
    """Load affine parameters from a JSON sidecar."""
    from .keypoint_register import AffineParams

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return AffineParams(**payload["affine_params"])
