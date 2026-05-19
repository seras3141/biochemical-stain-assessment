"""CZI loading helpers built on aicsimageio and its aicspylibczi backend."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from aicsimageio import AICSImage
from aicsimageio.readers import CziReader
from aicspylibczi import CziFile


def _ensure_aics_czi_compat() -> None:
    """Patch old aicsimageio against newer aicspylibczi, if needed."""
    if not hasattr(CziFile, "dims_shape") and hasattr(CziFile, "get_dims_shape"):
        CziFile.dims_shape = CziFile.get_dims_shape


def open_aics_image(path: str | Path) -> AICSImage:
    """Open a CZI with AICSImage using the CziReader/aicspylibczi backend."""
    _ensure_aics_czi_compat()
    return AICSImage(str(path), reader=CziReader)


def metadata_to_string(image: AICSImage) -> str:
    """Return AICSImage metadata as a unicode XML string."""
    metadata = image.metadata
    if isinstance(metadata, str):
        return metadata
    return ET.tostring(metadata, encoding="unicode")


def parse_czi_metadata(meta_str: str) -> tuple[float, float, float | None, float | None]:
    """Return (px_x_um, px_y_um, cx_um, cy_um) from CZI XML metadata."""
    hx = re.findall(
        r'<Distance Id="X">.*?<Value>([\d.eE+\-]+)</Value>',
        meta_str,
        re.DOTALL,
    )
    hy = re.findall(
        r'<Distance Id="Y">.*?<Value>([\d.eE+\-]+)</Value>',
        meta_str,
        re.DOTALL,
    )
    px_x = float(hx[0]) * 1e6
    px_y = float(hy[0]) * 1e6

    centres = re.findall(r"<CenterPosition>([\d.,\-]+)</CenterPosition>", meta_str)
    cx, cy = map(float, centres[0].split(",")) if centres else (None, None)

    return px_x, px_y, cx, cy


def _read_pixels_with_backend(path: str | Path) -> np.ndarray:
    """
    Read pixels through aicspylibczi, the backend used by aicsimageio for CZI.

    AICSImage 3.x exposes stitched mosaic CZIs as individual M tiles. For this
    project we need the stitched field of view, so mosaic files are read with
    read_mosaic while metadata and reader selection still go through AICSImage.
    """
    _ensure_aics_czi_compat()
    czi = CziFile(str(path))

    if czi.is_mosaic():
        read_kwargs = {"C": 0} if "C" in czi.dims else {}
        arr = czi.read_mosaic(**read_kwargs)
    else:
        read_kwargs = {"C": 0} if "C" in czi.dims else {}
        arr, _ = czi.read_image(**read_kwargs)

    arr = np.squeeze(arr)

    if arr.ndim == 3 and arr.shape[-1] == 1:
        arr = arr[:, :, 0]

    pixel_type = czi.pixel_type
    if isinstance(pixel_type, bytes):
        pixel_type = pixel_type.decode("utf-8", errors="ignore")
    if arr.ndim >= 3 and arr.shape[-1] in (3, 4) and "bgr" in str(pixel_type).lower():
        if arr.shape[-1] == 3:
            arr = arr[..., ::-1]
        else:
            arr = arr[..., [2, 1, 0, 3]]

    return arr


def read_czi(path: str | Path):
    """
    Open a CZI and return:
        img      - ndarray, usually (H, W), (H, W, 3), or a squeezed Z stack
        px_x_um  - pixel width in micrometres
        px_y_um  - pixel height in micrometres
        cx_um    - stage centre X in micrometres, when present
        cy_um    - stage centre Y in micrometres, when present
        meta_str - raw metadata XML string
    """
    image = open_aics_image(path)
    meta_str = metadata_to_string(image)
    arr = _read_pixels_with_backend(path)
    px_x, px_y, cx, cy = parse_czi_metadata(meta_str)
    return arr, px_x, px_y, cx, cy, meta_str
