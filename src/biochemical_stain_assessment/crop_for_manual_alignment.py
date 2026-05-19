#!/usr/bin/env python3
"""
crop_for_manual_alignment.py
-----------------------------
Saves each AbPAS / DAPI pair as two separate TIFF files at the same pixel size
and (approximate) field of view, ready for manual alignment in Fiji or ZEN.

Both output files will have:
  • Identical pixel size  (AbPAS native resolution: ~0.173 µm/px)
  • Identical canvas size (AbPAS dimensions: ~11 305 × 11 302 px)
  • Pixel calibration embedded as ImageJ-compatible TIFF metadata

Once open in Fiji:
  1. Open both TIFFs
  2. Plugins → Registration → Manual (or use the "Big Warp" plugin for
     landmark-based alignment)
  3. Alternatively: Image → Color → Merge Channels and adjust XY offset
     interactively with the "Align slices in stack" macro

Usage
-----
  python -m biochemical_stain_assessment.crop_for_manual_alignment
  python -m biochemical_stain_assessment.crop_for_manual_alignment <data_dir> <out_dir>

Output files per pair (in <out_dir>/separate/)
-----------------------------------------------
  <sample_id>_AbPAS.tif   – RGB brightfield, uint8
  <sample_id>_DAPI.tif    – fluorescence, uint16  (DAPI cropped to AbPAS FOV)
"""

import glob
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import zoom as ndzoom

from .czi_io import read_czi

warnings.filterwarnings("ignore")


# ── helpers ───────────────────────────────────────────────────────────────────


def resample(img, src_px, tgt_px):
    factor = src_px / tgt_px
    if abs(factor - 1.0) < 1e-4:
        return img
    src_dtype = img.dtype
    img_f = img.astype(np.float32)
    out = ndzoom(img_f, factor, order=1)
    info = np.iinfo(src_dtype) if np.issubdtype(src_dtype, np.integer) \
           else np.finfo(src_dtype)
    return np.clip(out, info.min, info.max).astype(src_dtype)


def crop_dapi_to_fov(dapi_rs, h_a, w_a, cx_a, cy_a, cx_d, cy_d, px):
    h_dr, w_dr = dapi_rs.shape[:2]
    dx_px =  (cx_a - cx_d) / px
    dy_px = -(cy_a - cy_d) / px
    r0 = int(round(h_dr / 2 + dy_px - h_a / 2))
    c0 = int(round(w_dr / 2 + dx_px - w_a / 2))
    r0 = max(0, r0)
    c0 = max(0, c0)
    r1 = min(h_dr, r0 + h_a)
    c1 = min(w_dr, c0 + w_a)
    crop = dapi_rs[r0:r1, c0:c1]
    # Pad to exact AbPAS size if edge clipping occurred
    pr = h_a - crop.shape[0]
    pc = w_a - crop.shape[1]
    if pr > 0 or pc > 0:
        crop = np.pad(crop, [(pr // 2, pr - pr // 2),
                             (pc // 2, pc - pc // 2)])
    return crop


# ── save as plain ImageJ TIFF with pixel calibration ──────────────────────────

def save_tiff(path, arr, px_um, channel_name):
    """
    Write a single-channel or RGB TIFF with ImageJ-compatible pixel size
    metadata (so Fiji shows the correct scale bar without Bio-Formats).
    """
    # ImageJ resolution stored as pixels-per-unit in the TIFF XResolution tag
    # unit = 3 (centimetre); 1 µm = 1e-4 cm → px/cm = 1e4 / px_um
    resolution = (1e4 / px_um, 1e4 / px_um)   # (x, y) pixels per centimetre

    if arr.ndim == 2:
        # Grayscale: tifffile expects (H, W)
        tifffile.imwrite(
            path,
            arr,
            resolution=resolution,
            resolutionunit=tifffile.RESUNIT.CENTIMETER,
            compression='deflate',
            compressionargs={'level': 6},
            metadata={'axes': 'YX'},
        )
    else:
        # RGB: tifffile expects (H, W, 3), photometric=rgb
        tifffile.imwrite(
            path,
            arr,
            photometric='rgb',
            resolution=resolution,
            resolutionunit=tifffile.RESUNIT.CENTIMETER,
            compression='deflate',
            compressionargs={'level': 6},
            metadata={'axes': 'YXS'},
        )
    print(f"    → {os.path.basename(path)}  "
          f"({arr.shape}, {arr.dtype}, {px_um:.4f} µm/px)")


# ── pairing ────────────────────────────────────────────────────────────────────

def find_pairs(data_dir):
    abpas_files = glob.glob(
        os.path.join(data_dir, '**', 'AbPAS', '*.czi'), recursive=True)
    pairs = []
    for ap in sorted(abpas_files):
        m = re.search(r'_AbPAS_(.+?)\.czi$', os.path.basename(ap))
        if not m:
            continue
        sample_id = m.group(1)
        parent = os.path.dirname(os.path.dirname(ap))
        matches = glob.glob(os.path.join(parent, 'DAPI', f'*_{sample_id}.czi'))
        if matches:
            pairs.append((sample_id, ap, matches[0]))
        else:
            print(f"  ⚠  No DAPI match for '{sample_id}' — skipping.")
    return pairs


# ── main ──────────────────────────────────────────────────────────────────────

def process_pair(sample_id, abpas_path, dapi_path, out_dir):
    print(f"\n  {sample_id}")

    # Read
    abpas, px_a, _, cx_a, cy_a, _ = read_czi(abpas_path)
    dapi,  px_d, _, cx_d, cy_d, _ = read_czi(dapi_path)
    h_a, w_a = abpas.shape[:2]

    # Resample DAPI to AbPAS pixel size
    dapi_rs = resample(dapi, px_d, px_a)

    # Crop DAPI to AbPAS field of view using stage coordinates
    dapi_crop = crop_dapi_to_fov(dapi_rs, h_a, w_a, cx_a, cy_a,
                                  cx_d, cy_d, px_a)

    # Save both at the same pixel size
    os.makedirs(out_dir, exist_ok=True)
    save_tiff(os.path.join(out_dir, f"{sample_id}_AbPAS.tif"),
              abpas, px_a, "AbPAS")
    save_tiff(os.path.join(out_dir, f"{sample_id}_DAPI.tif"),
              dapi_crop, px_a, "DAPI")


def main():
    data_dir = sys.argv[1] if len(sys.argv) > 1 else str(Path.cwd())
    out_dir  = sys.argv[2] if len(sys.argv) > 2 else \
        os.path.join(data_dir, "registered_output", "separate")

    print(f"Scanning: {data_dir}")
    print(f"Output:   {out_dir}")

    pairs = find_pairs(data_dir)
    if not pairs:
        print("No pairs found.")
        return

    print(f"\nFound {len(pairs)} pair(s) — resampling DAPI and cropping to AbPAS FOV:")
    for sample_id, ap, dp in pairs:
        process_pair(sample_id, ap, dp, out_dir)

    print(f"\n{'─'*55}")
    print(f"Done.  Files saved to: {out_dir}")
    print()
    print("To align manually in Fiji:")
    print("  1. File → Open both  *_AbPAS.tif  and  *_DAPI.tif")
    print("  2. Image → Color → Merge Channels")
    print("     (AbPAS → C2 green or C4 gray,  DAPI → C1 red or C3 blue)")
    print("  3. Use  Edit → Selection → Specify  or the")
    print("     Plugins → Registration → StackReg  plugin to fine-tune.")
    print()
    print("Both files share identical canvas size and pixel calibration,")
    print("so pixel (0,0) is the same physical location in both.")


if __name__ == "__main__":
    main()
