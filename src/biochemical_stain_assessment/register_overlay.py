#!/usr/bin/env python3
"""
register_overlay.py
-------------------
Registers AbPAS (brightfield RGB) and DAPI/Hoechst (fluorescence) CZI image
pairs acquired on different microscope cameras. For each matched pair it:

  1. Reads both CZI files and extracts pixel sizes + stage centre coordinates
     from the embedded OME-XML metadata.
  2. Resamples the DAPI image to match the AbPAS pixel size (≈ x2 upscale).
  3. Crops the resampled DAPI to the AbPAS field of view using stage coordinates.
  4. Normalises the AB-PAS image with Macenko stain normalisation.
  5. Enhances DAPI contrast with CLAHE to sharpen nuclear borders.
  6. Segments the organoid contour in both images and fits a scale + translation
     transform from ellipse geometry (falls back to stage-coordinate crop if it fails).
  7. Saves a 4-channel OME-TIFF  (R, G, B from AbPAS  +  DAPI)  with
     pixel-size metadata intact — ready to open in Fiji / ZEISS ZEN.
  8. Saves a quick composite PNG preview (AbPAS colour + DAPI in cyan).
  9. Saves an affine-params JSON sidecar alongside the OME-TIFF.

WHY WE CROP THIS WAY
---------------------
The two cameras have different pixel pitches (3.45 µm vs 6.9 µm) on the same
20x objective, giving:
  • AbPAS : 0.173 µm/px  →  11 305 x 11 302 px  (≈ 1 951 x 1 951 µm)
  • DAPI  : 0.344 µm/px  →   5 739 x 6 915 px   (≈ 1 975 x 2 380 µm)
After resampling DAPI to 0.173 µm/px the field becomes ≈ 11 445 x 13 790 px —
larger than AbPAS in both dimensions.  The stage XY centre coordinates stored in
each CZI tell us exactly where the AbPAS rectangle sits inside the larger DAPI
canvas, so we can crop to their intersection without guessing.

Usage
-----
  python -m biochemical_stain_assessment.register_overlay        # uses cwd
  python -m biochemical_stain_assessment.register_overlay <data_dir> <out_dir>

Output files per pair
---------------------
  <out_dir>/<sample_id>_registered.ome.tif   - 4-channel OME-TIFF
  <out_dir>/<sample_id>_preview.png           - composite RGB preview
"""

import glob
import logging
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import zoom as ndzoom
from skimage.registration import phase_cross_correlation
from skimage.transform import resize as sk_resize

from .czi_io import read_czi
from .registration import (
    AffineParams,
    abpas_inv_gray,
    apply_scale_translation,
    enhance_dapi,
    load_stain_matrix,
    mutual_crop,
    normalise_abpas,
    register_contour,
    save_affine_params,
)

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)

# ─────────────────────────────── helpers ──────────────────────────────────────


def _abpas_inv_gray(abpas_norm: np.ndarray) -> np.ndarray:
    """Convert normalised AB-PAS RGB to border-masked inverted luminance.

    Thin wrapper around :func:`registration.abpas_inv_gray` kept here so that
    ``process_pair`` does not need to be changed.
    """
    return abpas_inv_gray(abpas_norm)


# ─────────────────────────────── core steps ───────────────────────────────────


def resample_to_pixel_size(img: np.ndarray, src_px: float, tgt_px: float) -> np.ndarray:
    """
    Resample *img* from *src_px* µm/px to *tgt_px* µm/px using bilinear
    interpolation.  Works for 2-D (grayscale) and 3-D (H, W, C) arrays.
    """
    factor = src_px / tgt_px
    if abs(factor - 1.0) < 1e-4:
        return img

    src_dtype = img.dtype
    img_f = img.astype(np.float32)

    if img_f.ndim == 2:
        out = ndzoom(img_f, factor, order=1)
    else:  # (H, W, C)
        out = ndzoom(img_f, (factor, factor, 1), order=1)

    # Clip to original dtype range and convert back
    info = (
        np.iinfo(src_dtype)
        if np.issubdtype(src_dtype, np.integer)
        else np.finfo(src_dtype)
    )
    return np.clip(out, info.min, info.max).astype(src_dtype)


def crop_to_abpas_fov(
    dapi_rs: np.ndarray,
    h_abpas: int,
    w_abpas: int,
    cx_abpas: float,
    cy_abpas: float,
    cx_dapi: float,
    cy_dapi: float,
    tgt_px: float,
):
    """
    Crop *dapi_rs* (already resampled to *tgt_px* µm/px) to the rectangle
    that corresponds to the AbPAS field of view.

    Stage coordinates:
      • X increases rightward  → maps to image column direction  (same sign)
      • Y increases upward     → maps to image row direction      (inverted)

    Returns
    -------
    cropped : ndarray
    row_offset, col_offset : int   (top-left corner of the crop in dapi_rs;
                                    useful for debugging / visualisation)
    """
    h_dapi_rs, w_dapi_rs = dapi_rs.shape[:2]

    # AbPAS centre expressed in resampled-DAPI pixel coordinates
    dx_px = (cx_abpas - cx_dapi) / tgt_px
    dy_px = -(cy_abpas - cy_dapi) / tgt_px  # invert Y for image rows

    dapi_cy_px = h_dapi_rs / 2.0
    dapi_cx_px = w_dapi_rs / 2.0

    abpas_cy_in_dapi = dapi_cy_px + dy_px
    abpas_cx_in_dapi = dapi_cx_px + dx_px

    r0 = int(round(abpas_cy_in_dapi - h_abpas / 2.0))
    r1 = r0 + h_abpas
    c0 = int(round(abpas_cx_in_dapi - w_abpas / 2.0))
    c1 = c0 + w_abpas

    # Guard against floating-point rounding putting us 1 px out of bounds
    r0 = max(0, r0)
    r1 = min(h_dapi_rs, r1)
    c0 = max(0, c0)
    c1 = min(w_dapi_rs, c1)

    return dapi_rs[r0:r1, c0:c1], r0, c0


def refine_registration(
    abpas_gray: np.ndarray,
    dapi_cropped: np.ndarray,
    max_shift_px: int = 50,
    downsample_factor: int = 4,
) -> tuple:
    """
    Phase cross-correlation on downsampled images to find any residual
    offset remaining after the coordinate-based crop.

    AbPAS is brightfield (nuclei = dark on bright background) and DAPI is
    fluorescence (nuclei = bright on dark background), so we invert AbPAS
    before correlating.  The two modalities are very different in appearance,
    so cross-correlation can produce spurious large peaks — any detected shift
    larger than *max_shift_px* (default 50 px ≈ 8.6 µm) is discarded and the
    coordinate-based alignment is kept as-is.

    Returns (shift_row, shift_col) in FULL-resolution pixels.
    """
    f = downsample_factor
    target_shape = (abpas_gray.shape[0] // f, abpas_gray.shape[1] // f)

    ref = sk_resize(abpas_gray.astype(np.float32), target_shape, anti_aliasing=True)
    mov = sk_resize(dapi_cropped.astype(np.float32), target_shape, anti_aliasing=True)

    # Normalise each image to [0, 1] before inverting
    ref = (ref - ref.min()) / (ref.max() - ref.min() + 1e-9)
    mov = (mov - mov.min()) / (mov.max() - mov.min() + 1e-9)

    # Invert AbPAS so nuclei become bright (matching DAPI)
    ref_inv = 1.0 - ref

    shift_ds, _, _ = phase_cross_correlation(
        ref_inv, mov, upsample_factor=4, normalization=None
    )
    shift_full = (float(shift_ds[0]) * f, float(shift_ds[1]) * f)

    # Reject implausibly large shifts (stage coords are accurate to ~5 µm)
    if abs(shift_full[0]) > max_shift_px or abs(shift_full[1]) > max_shift_px:
        logger.warning(
            "Cross-correlation shift %s > %d px — trusting stage coordinates instead.",
            shift_full,
            max_shift_px,
        )
        return (0.0, 0.0)

    logger.info("Residual shift: (%+.1f, %+.1f) px", shift_full[0], shift_full[1])
    return shift_full


def apply_shift_and_crop(
    arr: np.ndarray, shift_row: float, shift_col: float
) -> np.ndarray:
    """
    Apply integer-pixel shift to *arr* by cropping (no interpolation) and
    return a view of the same shape.  Shifts the image by cutting from the
    leading or trailing edge.
    """
    dr = int(round(shift_row))
    dc = int(round(shift_col))
    h, w = arr.shape[:2]

    r0 = max(0, dr)
    r1 = min(h, h + dr)
    c0 = max(0, dc)
    c1 = min(w, w + dc)
    cropped = arr[r0:r1, c0:c1]

    # Pad back to original size with zeros (keeps spatial reference)
    if cropped.shape[:2] != (h, w):
        pad = [(max(0, -dr), max(0, dr)), (max(0, -dc), max(0, dc))]
        if arr.ndim == 3:
            pad.append((0, 0))
        cropped = np.pad(cropped, pad, mode="constant")

    return cropped[:h, :w]


# ─────────────────────────────── output writers ───────────────────────────────


def save_ome_tiff(
    path: str, abpas_rgb: np.ndarray, dapi: np.ndarray, px_um: float, sample_id: str
):
    """
    Save a 4-channel OME-TIFF (R, G, B, DAPI) with pixel-size metadata.
    Shape written: (C=4, Y, X).  DAPI is kept as uint16; RGB channels as uint8.
    """
    h, w = abpas_rgb.shape[:2]

    # Build channel array: (4, H, W)
    r = abpas_rgb[:, :, 0]
    g = abpas_rgb[:, :, 1]
    b = abpas_rgb[:, :, 2]

    # Normalise DAPI to uint16 full range for display
    d = dapi.astype(np.float32)
    d_norm = ((d - d.min()) / (d.max() - d.min() + 1e-9) * 65535).astype(np.uint16)

    # Store R/G/B as uint16 too so all channels share a dtype
    r16 = r.astype(np.uint16) * 257  # 255→65535
    g16 = g.astype(np.uint16) * 257
    b16 = b.astype(np.uint16) * 257

    stack = np.stack([r16, g16, b16, d_norm], axis=0)  # (4, H, W)

    # OME-XML metadata
    channel_names = ["AbPAS_R", "AbPAS_G", "AbPAS_B", "DAPI"]
    channel_colors = [
        0xFF0000FF,  # red  (RGBA, big-endian)
        0x00FF00FF,  # green
        0x0000FFFF,  # blue
        0x00FFFFFF,  # cyan  for DAPI
    ]

    metadata = {
        "axes": "CYX",
        "PhysicalSizeX": px_um,
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": px_um,
        "PhysicalSizeYUnit": "µm",
        "Channel": {
            "Name": channel_names,
            "Color": channel_colors,
        },
    }

    tifffile.imwrite(
        path,
        stack,
        photometric="minisblack",
        metadata=metadata,
        compression="deflate",
        compressionargs={"level": 6},
        imagej=False,
        ome=True,
    )
    logger.info("Saved OME-TIFF:  %s", os.path.basename(path))


def save_preview_png(
    path: str, abpas_rgb: np.ndarray, dapi: np.ndarray, downsample: int = 4
):
    """
    Save a composite RGB preview PNG at 1/4 resolution:
      • AbPAS colour image as background
      • DAPI shown in cyan, blended at 40 % opacity
    """
    from PIL import Image

    factor = 1.0 / downsample
    h, w = abpas_rgb.shape[:2]
    small_h, small_w = int(h * factor), int(w * factor)

    abpas_small = sk_resize(
        abpas_rgb.astype(np.float32) / 255.0, (small_h, small_w), anti_aliasing=True
    )

    dapi_f = dapi.astype(np.float32)
    dapi_norm = (dapi_f - dapi_f.min()) / (dapi_f.max() - dapi_f.min() + 1e-9)
    dapi_small = sk_resize(dapi_norm, (small_h, small_w), anti_aliasing=True)

    # Cyan overlay: add DAPI as (0, DAPI, DAPI)
    alpha = 0.40
    composite = abpas_small.copy()
    composite[:, :, 0] = np.clip(composite[:, :, 0] * (1 - alpha), 0, 1)
    composite[:, :, 1] = np.clip(composite[:, :, 1] + alpha * dapi_small, 0, 1)
    composite[:, :, 2] = np.clip(composite[:, :, 2] + alpha * dapi_small, 0, 1)

    img_uint8 = (composite * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img_uint8).save(path, optimize=True)
    logger.info("Saved preview:   %s", os.path.basename(path))


# ──────────────────────────────── pairing ─────────────────────────────────────


def find_pairs(data_dir: str):
    """
    Scan *data_dir* recursively and return a list of
    (sample_id, abpas_path, dapi_path) tuples.

    Naming convention assumed:
      AbPAS: *_AbPAS_<sample_id>.czi   (inside an AbPAS/ folder)
      DAPI:  *_Hoechst*_<sample_id>.czi  (inside a DAPI/ folder)
    """
    abpas_files = glob.glob(
        os.path.join(data_dir, "**", "AbPAS", "*.czi"), recursive=True
    )
    pairs = []
    for ap in sorted(abpas_files):
        # Extract sample ID: everything after the last underscore-separated stain token
        m = re.search(r"_AbPAS_(.+?)\.czi$", os.path.basename(ap))
        if not m:
            continue
        sample_id = m.group(1)

        # Find matching DAPI file (same parent-of-parent folder, same sample_id)
        parent = os.path.dirname(os.path.dirname(ap))  # e.g. .../JH-311
        dapi_pattern = os.path.join(parent, "DAPI", f"*_{sample_id}.czi")
        matches = glob.glob(dapi_pattern)
        if not matches:
            logger.warning("No DAPI match found for sample '%s' — skipping.", sample_id)
            continue
        pairs.append((sample_id, ap, matches[0]))

    return pairs


# ──────────────────────────────── main ────────────────────────────────────────


def process_pair(
    sample_id: str,
    abpas_path: str,
    dapi_path: str,
    out_dir: str,
    stain_matrix_path: str | None = None,
    refine: bool = True,
):
    """Full registration + export pipeline for one AbPAS / DAPI pair."""

    logger.info("─" * 60)
    logger.info("Sample: %s", sample_id)
    logger.info("AbPAS : %s", os.path.basename(abpas_path))
    logger.info("DAPI  : %s", os.path.basename(dapi_path))

    # ── 1. Read ──────────────────────────────────────────────────────────────
    logger.info("[1/7] Reading CZI files …")
    abpas, px_abpas, _, cx_abpas, cy_abpas, _ = read_czi(abpas_path)
    dapi, px_dapi, _, cx_dapi, cy_dapi, _ = read_czi(dapi_path)

    h_abpas, w_abpas = abpas.shape[:2]
    logger.debug(
        "AbPAS %s  px=%.4f µm  centre=(%.1f, %.1f)",
        abpas.shape,
        px_abpas,
        cx_abpas,
        cy_abpas,
    )
    logger.debug(
        "DAPI  %s  px=%.4f µm  centre=(%.1f, %.1f)",
        dapi.shape,
        px_dapi,
        cx_dapi,
        cy_dapi,
    )

    # ── 2. Resample DAPI ─────────────────────────────────────────────────────
    scale = px_dapi / px_abpas
    logger.info("[2/7] Resampling DAPI x%.4f to %.4f µm/px …", scale, px_abpas)
    dapi_rs = resample_to_pixel_size(dapi, px_dapi, px_abpas)
    logger.debug("Resampled DAPI: %s", dapi_rs.shape)

    # ── 3. Coordinate-based crop ─────────────────────────────────────────────
    logger.info("[3/7] Cropping DAPI to AbPAS field of view (stage coordinates) …")
    dapi_crop, r0, c0 = crop_to_abpas_fov(
        dapi_rs, h_abpas, w_abpas, cx_abpas, cy_abpas, cx_dapi, cy_dapi, px_abpas
    )
    logger.debug("Crop origin in resampled DAPI: row=%d, col=%d", r0, c0)
    logger.debug("Cropped DAPI: %s", dapi_crop.shape)

    # If crop came out slightly short (edge rounding), pad symmetrically
    if dapi_crop.shape[0] != h_abpas or dapi_crop.shape[1] != w_abpas:
        pad_r = h_abpas - dapi_crop.shape[0]
        pad_c = w_abpas - dapi_crop.shape[1]
        dapi_crop = np.pad(
            dapi_crop,
            [(pad_r // 2, pad_r - pad_r // 2), (pad_c // 2, pad_c - pad_c // 2)],
            mode="constant",
        )
        logger.debug("Padded to: %s", dapi_crop.shape)

    # ── 4-6. Content-based contour refinement ────────────────────────────────
    abpas_final = abpas
    dapi_final = dapi_crop
    params = AffineParams(
        scale=1.0,
        tx=0.0,
        ty=0.0,
        n_inliers=0,
        method="stage_coords",
        rmse=0.0,
    )
    if refine:
        logger.info("[4/7] Normalising AB-PAS staining (Macenko) …")
        stain_matrix = (
            load_stain_matrix(stain_matrix_path) if stain_matrix_path else None
        )
        abpas_norm = normalise_abpas(abpas, stain_matrix_target=stain_matrix)

        logger.info("[5/7] Enhancing DAPI contrast (CLAHE) …")
        dapi_enhanced = enhance_dapi(dapi_crop)

        logger.info("[6/7] Registering with contours (scale + translation) …")
        fallback = AffineParams(
            scale=1.0,
            tx=0.0,
            ty=0.0,
            n_inliers=0,
            method="stage_coords",
            rmse=0.0,
        )
        ref_gray = _abpas_inv_gray(abpas_norm)
        params, _, _ = register_contour(
            ref_gray,
            dapi_enhanced,
            px_um=px_abpas,
            max_shift_um=100.0,
        )
        if params.method == "contour_failed":
            logger.warning(
                "Contour registration failed; using stage-coordinate crop fallback."
            )
            params = fallback

        logger.info(
            "%s: scale=%.5f  tx=%+.1f  ty=%+.1f  inliers=%d  rmse=%.2f px",
            params.method,
            params.scale,
            params.tx,
            params.ty,
            params.n_inliers,
            params.rmse,
        )
        dapi_aligned = apply_scale_translation(
            dapi_enhanced,
            params.scale,
            params.tx,
            params.ty,
            output_shape=abpas_norm.shape[:2],
            order=1,
        )
        abpas_final, dapi_final = mutual_crop(abpas_norm, dapi_aligned, margin=64)
    else:
        logger.info("[4/7] Skipping contour refinement; using stage-coordinate crop.")
        logger.info("[5/7] Skipping DAPI contrast enhancement.")
        logger.info("[6/7] Skipping mutual crop.")

    # ── 7. Save outputs ──────────────────────────────────────────────────────
    logger.info("[7/7] Saving outputs …")
    os.makedirs(out_dir, exist_ok=True)
    tiff_path = os.path.join(out_dir, f"{sample_id}_registered.ome.tif")
    png_path = os.path.join(out_dir, f"{sample_id}_preview.png")
    params_path = os.path.join(out_dir, f"{sample_id}_affine_params.json")

    save_ome_tiff(tiff_path, abpas_final, dapi_final, px_abpas, sample_id)
    save_preview_png(png_path, abpas_final, dapi_final)
    save_affine_params(params, params_path, sample_id, abpas_path, dapi_path)
    logger.info("Saved affine:    %s", os.path.basename(params_path))


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir = sys.argv[1] if len(sys.argv) > 1 else str(Path.cwd())
    out_dir = (
        sys.argv[2]
        if len(sys.argv) > 2
        else os.path.join(data_dir, "registered_output")
    )

    logger.info("Scanning for AbPAS/DAPI pairs in:  %s", data_dir)
    logger.info("Output directory:                  %s", out_dir)

    pairs = find_pairs(data_dir)
    if not pairs:
        logger.warning("No pairs found — check folder structure and naming convention.")
        return

    logger.info("Found %d pair(s):", len(pairs))
    for sid, _ap, _dp in pairs:
        logger.info("  %s", sid)

    for sample_id, abpas_path, dapi_path in pairs:
        process_pair(sample_id, abpas_path, dapi_path, out_dir, refine=True)

    logger.info("═" * 60)
    logger.info("Done.  Results in: %s", out_dir)
    logger.info("Open the .ome.tif files in Fiji → drag & drop → Bio-Formats importer")
    logger.info("Each file has 4 channels: AbPAS_R, AbPAS_G, AbPAS_B, DAPI")


if __name__ == "__main__":
    main()
