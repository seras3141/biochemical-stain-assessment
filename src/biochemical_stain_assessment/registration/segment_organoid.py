"""Organoid-level segmentation helpers for contour registration."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gaussian, threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import binary_closing, disk, remove_small_holes
from skimage.transform import downscale_local_mean

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OrganoidProps:
    """Geometric properties of a segmented organoid in input-image pixels."""

    centroid_row: float
    centroid_col: float
    axis_major: float
    axis_minor: float
    orientation: float
    area_px: int
    mean_axis: float


def abpas_inv_gray(
    abpas_norm: np.ndarray,
    valid_threshold: float = 0.01,
) -> np.ndarray:
    """Convert Macenko-normalised AB-PAS RGB to inverted float32 luminance.

    Black border regions produced by tile-stitching artefacts (pixels where all
    RGB channels are near zero) are zeroed in the output rather than being
    inverted to maximum brightness.  Without this masking, :func:`segment_organoid`
    mistakes the bright-after-inversion borders for organoid foreground, causing
    Otsu thresholding to fire on the border padding instead of the tissue, and
    producing a badly fitted ellipse that spans the stitching cross rather than
    the organoid.

    Parameters
    ----------
    abpas_norm : ndarray, shape (H, W, 3), dtype uint8
        Macenko-normalised AB-PAS image as returned by :func:`normalise_abpas`.
    valid_threshold : float
        Grayscale value (in [0, 1]) below which a pixel is considered background
        padding and is zeroed in the output.  Default 0.01 (i.e. pixels whose
        normalised luminance is ≤ 1 % are treated as black borders).

    Returns
    -------
    ndarray, shape (H, W), dtype float32, range [0, 1]
        Inverted luminance image with stitching-artefact borders zeroed out.

    Raises
    ------
    ValueError
        If *abpas_norm* is not a 3-D array with at least 3 channels.

    Notes
    -----
    The valid-pixel mask ``gray > valid_threshold`` is applied multiplicatively
    so that the output is zero wherever the input was black, and ``1 - gray``
    elsewhere.  The threshold of 0.01 was chosen empirically: genuine tissue
    pixels in Macenko-normalised AB-PAS have luminance > 0.05 even in the
    lightest areas, so the threshold leaves all real tissue unaffected while
    eliminating padding borders.
    """
    rgb = np.asarray(abpas_norm)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(
            f"abpas_norm must have shape (H, W, ≥3); got shape {rgb.shape}."
        )
    gray = rgb2gray(rgb[:, :, :3].astype(np.float32) / 255.0)
    inv = (1.0 - gray).astype(np.float32)
    valid = gray > valid_threshold
    return (inv * valid).astype(np.float32)


def segment_organoid(
    img: np.ndarray,
    smooth_sigma: float = 5.0,
    closing_radius: int = 20,
    min_hole_area: int = 5000,
    foreground_is_bright: bool = True,
    downsample_factor: int = 8,
) -> tuple[np.ndarray, OrganoidProps]:
    """Segment the organoid from background and return its mask and geometry.

    Parameters
    ----------
    img : ndarray, shape (H, W)
        Preprocessed single-channel image. Coordinates are pixel rows and
        columns in the input image.
    smooth_sigma : float
        Gaussian sigma in pixels before Otsu thresholding.
    closing_radius : int
        Disk radius for binary closing, in pixels.
    min_hole_area : int
        Maximum hole area to fill after closing, in pixels.
    foreground_is_bright : bool
        If True, pixels above the Otsu threshold are organoid foreground.
        If False, pixels below the threshold are organoid foreground.
    downsample_factor : int
        Factor by which to reduce the image before segmentation. Returned
        ``OrganoidProps`` coordinates are always in full-resolution pixels.
        Use 1 to skip downscaling for small synthetic images.

    Returns
    -------
    mask : ndarray, shape (H, W), dtype bool
        Binary organoid mask at the input image resolution.
    props : OrganoidProps
        Ellipse-like geometric properties from the largest connected component,
        expressed in full-resolution input pixels.

    Raises
    ------
    ValueError
        If ``img`` is not 2-D or no foreground component can be extracted.

    Notes
    -----
    Downscaling uses ``skimage.transform.downscale_local_mean``, which averages
    pixel blocks before thresholding. This suppresses individual nuclei and fine
    fibres while preserving the low-frequency organoid boundary used for ellipse
    fitting.
    """
    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("img must be a 2-D array.")
    if downsample_factor < 1:
        raise ValueError("downsample_factor must be at least 1.")

    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    if float(arr.max()) <= float(arr.min()):
        raise ValueError("Cannot segment a blank organoid image.")

    factor = int(downsample_factor)
    arr_work = (
        downscale_local_mean(arr, (factor, factor)).astype(np.float32)
        if factor > 1
        else arr
    )

    smoothed = gaussian(arr_work, sigma=smooth_sigma, preserve_range=True)
    valid = (
        smoothed > 0.01 if foreground_is_bright else smoothed < 0.99
    )  # Remove extreme values
    if not np.any(valid):
        mode = "bright" if foreground_is_bright else "dark"
        raise ValueError(
            f"Cannot determine an Otsu threshold: no valid pixels remain after "
            f"excluding extreme values for {mode}-foreground segmentation."
        )
    threshold = threshold_otsu(smoothed[valid])
    mask = smoothed > threshold if foreground_is_bright else smoothed < threshold

    if closing_radius > 0:
        mask = binary_closing(mask, disk(closing_radius))
    if min_hole_area > 0:
        mask = remove_small_holes(mask, area_threshold=min_hole_area)

    mask, region = _extract_largest_component(mask)
    props = OrganoidProps(
        centroid_row=float(region.centroid[0] * factor),
        centroid_col=float(region.centroid[1] * factor),
        axis_major=float(region.axis_major_length * factor),
        axis_minor=float(region.axis_minor_length * factor),
        orientation=float(region.orientation),
        area_px=int(region.area * factor**2),
        mean_axis=float(
            (region.axis_major_length + region.axis_minor_length) / 2.0 * factor
        ),
    )
    mask_full = _upsample_mask(mask, arr.shape, factor)
    logger.debug(
        "Segmented organoid: area=%d centroid=(%.2f, %.2f) mean_axis=%.2f "
        "downsample_factor=%d",
        props.area_px,
        props.centroid_row,
        props.centroid_col,
        props.mean_axis,
        factor,
    )
    return mask_full, props


def _extract_largest_component(mask: np.ndarray) -> tuple[np.ndarray, Any]:
    """Return the largest connected component mask and its regionprops object."""
    labels = label(mask)
    regions = regionprops(labels)
    if not regions:
        raise ValueError("No foreground organoid component found.")

    region = max(regions, key=lambda candidate: candidate.area)
    component = labels == region.label
    if int(region.area) == 0:
        raise ValueError("No non-empty organoid component found.")
    return component, region


def _upsample_mask(
    mask: np.ndarray,
    output_shape: tuple[int, int],
    factor: int,
) -> np.ndarray:
    """Expand a downsampled mask back to the input image shape."""
    if factor == 1:
        return mask.astype(bool, copy=False)

    upsampled = np.repeat(np.repeat(mask, factor, axis=0), factor, axis=1)
    return upsampled[: output_shape[0], : output_shape[1]].astype(bool, copy=False)
