"""Contour-based scale and translation registration."""

from __future__ import annotations

import logging

import numpy as np

from .keypoint_register import AffineParams
from .segment_organoid import OrganoidProps, segment_organoid

logger = logging.getLogger(__name__)


def register_contour(
    abpas_inv_gray: np.ndarray,
    dapi_enhanced: np.ndarray,
    smooth_sigma: float = 5.0,
    closing_radius: int = 20,
    min_hole_area: int = 5000,
    max_scale_deviation: float = 0.15,
    max_shift_um: float | None = None,
    px_um: float | None = None,
    downsample_factor: int = 8,
) -> tuple[AffineParams, OrganoidProps, OrganoidProps]:
    """Estimate scale and translation from matched organoid ellipse properties.

    Parameters
    ----------
    abpas_inv_gray : ndarray, shape (H, W), float32, range [0, 1]
        Inverted luminance of the normalised AB-PAS image.
    dapi_enhanced : ndarray, shape (H, W), float32, range [0, 1]
        CLAHE-enhanced DAPI image.
    smooth_sigma : float
        Gaussian sigma passed to ``segment_organoid``.
    closing_radius : int
        Binary closing radius passed to ``segment_organoid``.
    min_hole_area : int
        Hole-filling area threshold passed to ``segment_organoid``.
    max_scale_deviation : float
        Maximum allowed absolute deviation from scale 1.0.
    max_shift_um : float or None
        Optional maximum physical shift magnitude in microns.
    px_um : float or None
        Pixel size in microns per pixel. Required when ``max_shift_um`` is set.
    downsample_factor : int
        Factor passed to ``segment_organoid`` for parameter estimation. Returned
        contour properties remain in full-resolution pixel coordinates.

    Returns
    -------
    params : AffineParams
        ``method`` is ``"contour"`` on success or ``"contour_failed"`` on failure.
    props_ref : OrganoidProps
        Segmented AB-PAS organoid properties.
    props_mov : OrganoidProps
        Segmented DAPI organoid properties.
    """
    if max_shift_um is not None and px_um is None:
        raise ValueError("px_um is required when max_shift_um is set.")

    try:
        _, props_ref = segment_organoid(
            abpas_inv_gray,
            smooth_sigma=smooth_sigma,
            closing_radius=closing_radius,
            min_hole_area=min_hole_area,
            foreground_is_bright=True,
            downsample_factor=downsample_factor,
        )
        _, props_mov = segment_organoid(
            dapi_enhanced,
            smooth_sigma=smooth_sigma,
            closing_radius=closing_radius,
            min_hole_area=min_hole_area,
            foreground_is_bright=True,
            downsample_factor=downsample_factor,
        )
    except ValueError as exc:
        logger.warning("Contour registration segmentation failed: %s", exc)
        empty = _empty_props()
        return _failed_params(), empty, empty

    if props_mov.mean_axis <= 0.0:
        logger.warning("Contour registration failed: moving mean axis is zero.")
        return _failed_params(), props_ref, props_mov

    scale = props_ref.mean_axis / props_mov.mean_axis
    ty = props_ref.centroid_row - scale * props_mov.centroid_row
    tx = props_ref.centroid_col - scale * props_mov.centroid_col
    rmse = _ellipse_residual(props_ref, props_mov, scale, tx, ty)
    params = AffineParams(
        scale=float(scale),
        tx=float(tx),
        ty=float(ty),
        n_inliers=1,
        method="contour",
        rmse=rmse,
    )

    if abs(params.scale - 1.0) > max_scale_deviation:
        logger.warning(
            "Contour registration: scale=%.4f outside plausible range [%.2f, %.2f].",
            params.scale,
            1.0 - max_scale_deviation,
            1.0 + max_scale_deviation,
        )
        return (
            _failed_params(scale=params.scale, tx=params.tx, ty=params.ty, rmse=rmse),
            props_ref,
            props_mov,
        )

    if max_shift_um is not None and px_um is not None:
        shift_um = float(np.hypot(params.tx, params.ty) * px_um)
        if shift_um > max_shift_um:
            logger.warning(
                "Contour registration: shift=%.1f µm > max_shift_um=%.1f µm.",
                shift_um,
                max_shift_um,
            )
            return (
                _failed_params(
                    scale=params.scale,
                    tx=params.tx,
                    ty=params.ty,
                    rmse=rmse,
                ),
                props_ref,
                props_mov,
            )

    logger.info(
        "Contour registration succeeded: scale=%.5f tx=%+.1f ty=%+.1f rmse=%.2f px",
        params.scale,
        params.tx,
        params.ty,
        params.rmse,
    )
    return params, props_ref, props_mov



def register_contour_from_props(
    props_abpas: OrganoidProps,
    props_dapi: OrganoidProps,
    max_scale_deviation: float = 0.15,
    max_shift_um: float | None = None,
    px_um: float | None = None,
    downsample_factor: int = 8,
) -> tuple[AffineParams, OrganoidProps, OrganoidProps]:
    """Estimate scale and translation from matched organoid ellipse properties.

    Parameters
    ----------
    props_abpas : OrganoidProps
        Segmented AB-PAS organoid properties.
    props_dapi : OrganoidProps
        Segmented DAPI organoid properties.
    max_scale_deviation : float
        Maximum allowed absolute deviation from scale 1.0.
    max_shift_um : float or None
        Optional maximum physical shift magnitude in microns.
    px_um : float or None
        Pixel size in microns per pixel. Required when ``max_shift_um`` is set.
    downsample_factor : int
        Factor passed to ``segment_organoid`` for parameter estimation. Returned
        contour properties remain in full-resolution pixel coordinates.

    Returns
    -------
    params : AffineParams
        ``method`` is ``"contour"`` on success or ``"contour_failed"`` on failure.
    props_ref : OrganoidProps
        Segmented AB-PAS organoid properties.
    props_mov : OrganoidProps
        Segmented DAPI organoid properties.
    """
    if max_shift_um is not None and px_um is None:
        raise ValueError("px_um is required when max_shift_um is set.")
    
    props_ref = props_abpas
    props_mov = props_dapi

    if props_mov.mean_axis <= 0.0:
        logger.warning("Contour registration failed: moving mean axis is zero.")
        return _failed_params(), props_ref, props_mov

    scale = props_ref.mean_axis / props_mov.mean_axis
    ty = props_ref.centroid_row - scale * props_mov.centroid_row
    tx = props_ref.centroid_col - scale * props_mov.centroid_col
    rmse = _ellipse_residual(props_ref, props_mov, scale, tx, ty)
    params = AffineParams(
        scale=float(scale),
        tx=float(tx),
        ty=float(ty),
        n_inliers=1,
        method="contour",
        rmse=rmse,
    )

    if abs(params.scale - 1.0) > max_scale_deviation:
        logger.warning(
            "Contour registration: scale=%.4f outside plausible range [%.2f, %.2f].",
            params.scale,
            1.0 - max_scale_deviation,
            1.0 + max_scale_deviation,
        )
        return (
            _failed_params(scale=params.scale, tx=params.tx, ty=params.ty, rmse=rmse),
            props_ref,
            props_mov,
        )

    if max_shift_um is not None and px_um is not None:
        shift_um = float(np.hypot(params.tx, params.ty) * px_um)
        if shift_um > max_shift_um:
            logger.warning(
                "Contour registration: shift=%.1f µm > max_shift_um=%.1f µm.",
                shift_um,
                max_shift_um,
            )
            return (
                _failed_params(
                    scale=params.scale,
                    tx=params.tx,
                    ty=params.ty,
                    rmse=rmse,
                ),
                props_ref,
                props_mov,
            )

    logger.info(
        "Contour registration succeeded: scale=%.5f tx=%+.1f ty=%+.1f rmse=%.2f px",
        params.scale,
        params.tx,
        params.ty,
        params.rmse,
    )
    return params, props_ref, props_mov


def _ellipse_residual(
    props_ref: OrganoidProps,
    props_mov: OrganoidProps,
    scale: float,
    tx: float,
    ty: float,
) -> float:
    centroid_residual = np.hypot(
        props_ref.centroid_row - (scale * props_mov.centroid_row + ty),
        props_ref.centroid_col - (scale * props_mov.centroid_col + tx),
    )
    axis_residual = abs(props_ref.mean_axis - scale * props_mov.mean_axis)
    return float((centroid_residual + axis_residual) / 2.0)


def _failed_params(
    scale: float = 1.0,
    tx: float = 0.0,
    ty: float = 0.0,
    rmse: float = float("inf"),
) -> AffineParams:
    return AffineParams(
        scale=float(scale),
        tx=float(tx),
        ty=float(ty),
        n_inliers=0,
        method="contour_failed",
        rmse=float(rmse),
    )


def _empty_props() -> OrganoidProps:
    return OrganoidProps(
        centroid_row=0.0,
        centroid_col=0.0,
        axis_major=0.0,
        axis_minor=0.0,
        orientation=0.0,
        area_px=0,
        mean_axis=0.0,
    )
