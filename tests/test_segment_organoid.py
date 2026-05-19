import numpy as np
import pytest

from biochemical_stain_assessment.registration.segment_organoid import (
    abpas_inv_gray,
    segment_organoid,
)


def _circle_image(
    size: int = 256,
    cy: float = 128.0,
    cx: float = 128.0,
    radius: float = 80.0,
    foreground: float = 1.0,
    background: float = 0.0,
) -> np.ndarray:
    rr, cc = np.ogrid[:size, :size]
    img = np.full((size, size), background, dtype=np.float32)
    img[(rr - cy) ** 2 + (cc - cx) ** 2 <= radius**2] = foreground
    return img


def _ellipse_image(
    size: int = 256,
    cy: float = 128.0,
    cx: float = 128.0,
    axis_row: float = 50.0,
    axis_col: float = 75.0,
) -> np.ndarray:
    rr, cc = np.ogrid[:size, :size]
    img = np.zeros((size, size), dtype=np.float32)
    mask = ((rr - cy) / axis_row) ** 2 + ((cc - cx) / axis_col) ** 2 <= 1.0
    img[mask] = 1.0
    return img


def test_segments_circular_blob() -> None:
    img = _circle_image(radius=80)

    mask, props = segment_organoid(
        img, closing_radius=5, min_hole_area=1000, downsample_factor=1
    )

    expected_area = np.pi * 80**2
    assert mask.sum() > expected_area * 0.90
    assert abs(props.centroid_row - 128.0) < 2.0
    assert abs(props.centroid_col - 128.0) < 2.0


def test_segments_elliptical_blob() -> None:
    img = _ellipse_image(axis_row=50, axis_col=75)

    _, props = segment_organoid(
        img, closing_radius=5, min_hole_area=1000, downsample_factor=1
    )

    assert abs((props.axis_major / props.axis_minor) - 1.5) < 0.08


def test_fills_internal_holes() -> None:
    img = _circle_image(radius=80)
    rr, cc = np.ogrid[:256, :256]
    img[(rr - 128) ** 2 + (cc - 128) ** 2 <= 10**2] = 0.0

    mask, _ = segment_organoid(
        img, closing_radius=5, min_hole_area=1000, downsample_factor=1
    )

    assert mask[128, 128]


def test_raises_on_blank_image() -> None:
    with pytest.raises(ValueError, match="blank"):
        segment_organoid(np.zeros((64, 64), dtype=np.float32), downsample_factor=1)


def test_foreground_is_dark() -> None:
    img = _circle_image(radius=80, foreground=0.0, background=1.0)

    mask, props = segment_organoid(
        img,
        closing_radius=5,
        min_hole_area=1000,
        foreground_is_bright=False,
        downsample_factor=1,
    )

    assert mask.sum() > np.pi * 80**2 * 0.90
    assert abs(props.centroid_row - 128.0) < 2.0
    assert abs(props.centroid_col - 128.0) < 2.0


# ──────────────────────────────── abpas_inv_gray ──────────────────────────────


def _abpas_rgb(h: int = 64, w: int = 64, value: int = 128) -> np.ndarray:
    """Return a flat uint8 RGB image with every pixel set to *value*."""
    return np.full((h, w, 3), value, dtype=np.uint8)


def test_abpas_inv_gray_black_borders_become_zero() -> None:
    """Pixels that are pure black in the input must be 0 in the output.

    This is the core fix: stitching-artefact borders (value 0 in the original
    CZI mosaic) invert naively to 1.0, causing segment_organoid to mistake them
    for bright organoid foreground.  The border-masking step must zero them out.
    """
    img = _abpas_rgb(value=128)
    # Set a 10-pixel border to black (simulating stitching padding)
    img[:10, :] = 0
    img[-10:, :] = 0
    img[:, :10] = 0
    img[:, -10:] = 0

    out = abpas_inv_gray(img)

    assert out.dtype == np.float32
    # Border region must be exactly zero
    assert np.all(out[:10, :] == 0.0), "top border must be zeroed"
    assert np.all(out[-10:, :] == 0.0), "bottom border must be zeroed"
    assert np.all(out[:, :10] == 0.0), "left border must be zeroed"
    assert np.all(out[:, -10:] == 0.0), "right border must be zeroed"
    # Interior (non-black) pixels must be inverted, not zero
    assert np.all(out[10:-10, 10:-10] > 0.0), "interior pixels must be > 0 after inversion"


def test_abpas_inv_gray_valid_pixels_are_inverted() -> None:
    """For pixels with non-zero input luminance, output == 1 - luminance."""
    # Pure mid-grey image (no borders)
    img = _abpas_rgb(value=128)
    out = abpas_inv_gray(img)

    from skimage.color import rgb2gray

    gray = rgb2gray(img.astype(np.float32) / 255.0)
    expected = (1.0 - gray).astype(np.float32)
    np.testing.assert_allclose(out, expected, atol=1e-6)


def test_abpas_inv_gray_output_range() -> None:
    """Output values must lie in [0, 1]."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
    out = abpas_inv_gray(img)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_abpas_inv_gray_raises_on_wrong_shape() -> None:
    """A 2-D array must raise ValueError."""
    with pytest.raises(ValueError, match="shape"):
        abpas_inv_gray(np.zeros((64, 64), dtype=np.uint8))


def test_abpas_inv_gray_custom_threshold() -> None:
    """valid_threshold is respected: pixels at or below it are zeroed."""
    img = _abpas_rgb(value=0)
    # Set interior to a value whose luminance is exactly at the boundary
    # rgb2gray of (10, 10, 10) ≈ 0.039, which is > default threshold 0.01
    img[10:20, 10:20] = 10
    out_default = abpas_inv_gray(img, valid_threshold=0.01)
    out_strict = abpas_inv_gray(img, valid_threshold=0.05)

    # With default threshold interior (luminance ≈ 0.039) should survive
    assert np.any(out_default[10:20, 10:20] > 0.0)
    # With strict threshold (0.05) the same pixels should be zeroed
    assert np.all(out_strict[10:20, 10:20] == 0.0)


# ──────────────────────────────── segment_organoid ────────────────────────────


def test_downscale_returns_fullres_coords() -> None:
    img = _circle_image(size=512, cy=256.0, cx=256.0, radius=120.0)

    mask, props = segment_organoid(
        img,
        closing_radius=5,
        min_hole_area=1000,
        downsample_factor=4,
    )

    tolerance = 4 * np.sqrt(2)
    assert mask.shape == img.shape
    assert abs(props.centroid_row - 256.0) < tolerance
    assert abs(props.centroid_col - 256.0) < tolerance
    assert abs(props.mean_axis - 240.0) < tolerance
