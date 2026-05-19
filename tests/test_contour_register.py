import numpy as np

from biochemical_stain_assessment.registration.contour_register import register_contour


def _circle_image(
    size: int = 512,
    cy: float = 256.0,
    cx: float = 256.0,
    radius: float = 150.0,
) -> np.ndarray:
    rr, cc = np.ogrid[:size, :size]
    img = np.zeros((size, size), dtype=np.float32)
    img[(rr - cy) ** 2 + (cc - cx) ** 2 <= radius**2] = 1.0
    return img


def _ellipse_image(
    size: int = 512,
    cy: float = 256.0,
    cx: float = 256.0,
    axis_row: float = 95.0,
    axis_col: float = 170.0,
) -> np.ndarray:
    rr, cc = np.ogrid[:size, :size]
    img = np.zeros((size, size), dtype=np.float32)
    mask = ((rr - cy) / axis_row) ** 2 + ((cc - cx) / axis_col) ** 2 <= 1.0
    img[mask] = 1.0
    return img


def test_identity() -> None:
    img = _circle_image()

    params, _, _ = register_contour(
        img, img, closing_radius=5, min_hole_area=1000, downsample_factor=1
    )

    assert params.method == "contour"
    assert abs(params.scale - 1.0) < 0.01
    assert abs(params.tx) < 1.0
    assert abs(params.ty) < 1.0
    assert params.rmse < 1.0


def test_known_translation() -> None:
    ref = _circle_image()
    mov = _circle_image(cy=271.0, cx=236.0)

    params, _, _ = register_contour(
        ref, mov, closing_radius=5, min_hole_area=1000, downsample_factor=1
    )

    assert params.method == "contour"
    assert abs(params.scale - 1.0) < 0.01
    assert abs(params.ty + 15.0) < 1.0
    assert abs(params.tx - 20.0) < 1.0


def test_known_scale() -> None:
    ref = _circle_image(radius=150.0)
    mov = _circle_image(radius=150.0 / 1.05)

    params, _, _ = register_contour(
        ref, mov, closing_radius=5, min_hole_area=1000, downsample_factor=1
    )

    assert params.method == "contour"
    assert abs(params.scale - 1.05) < 0.02


def test_known_scale_and_translation() -> None:
    expected_scale = 0.97
    expected_ty = 10.0
    expected_tx = 5.0
    ref = _circle_image(radius=145.0)
    mov = _circle_image(
        cy=(256.0 - expected_ty) / expected_scale,
        cx=(256.0 - expected_tx) / expected_scale,
        radius=145.0 / expected_scale,
    )

    params, _, _ = register_contour(
        ref, mov, closing_radius=5, min_hole_area=1000, downsample_factor=1
    )

    assert params.method == "contour"
    assert abs(params.scale - expected_scale) < 0.02
    assert abs(params.ty - expected_ty) < 2.0
    assert abs(params.tx - expected_tx) < 2.0


def test_flags_implausible_scale() -> None:
    ref = _circle_image(radius=160.0)
    mov = _circle_image(radius=80.0)

    params, _, _ = register_contour(
        ref,
        mov,
        closing_radius=5,
        min_hole_area=1000,
        max_scale_deviation=0.15,
        downsample_factor=1,
    )

    assert params.method == "contour_failed"
    assert params.n_inliers == 0


def test_flags_implausible_shift() -> None:
    ref = _circle_image(cy=256.0, cx=256.0)
    mov = _circle_image(cy=230.0, cx=256.0)

    params, _, _ = register_contour(
        ref,
        mov,
        closing_radius=5,
        min_hole_area=1000,
        max_shift_um=10.0,
        px_um=1.0,
        downsample_factor=1,
    )

    assert params.method == "contour_failed"
    assert params.n_inliers == 0


def test_elliptical_organoids() -> None:
    ref = _ellipse_image(axis_row=95.0, axis_col=170.0)
    mov = _ellipse_image(axis_row=95.0 / 1.04, axis_col=170.0 / 1.04)

    params, _, _ = register_contour(
        ref, mov, closing_radius=5, min_hole_area=1000, downsample_factor=1
    )

    assert params.method == "contour"
    assert abs(params.scale - 1.04) < 0.02


def test_register_contour_with_downscale() -> None:
    expected_scale = 1.03
    expected_ty = -16.0
    expected_tx = 24.0
    ref = _circle_image(size=1024, cy=512.0, cx=512.0, radius=260.0)
    mov = _circle_image(
        size=1024,
        cy=(512.0 - expected_ty) / expected_scale,
        cx=(512.0 - expected_tx) / expected_scale,
        radius=260.0 / expected_scale,
    )

    params, _, _ = register_contour(
        ref,
        mov,
        closing_radius=5,
        min_hole_area=1000,
        downsample_factor=4,
    )

    tolerance = 4 * np.sqrt(2)
    assert params.method == "contour"
    assert abs(params.scale - expected_scale) < 0.02
    assert abs(params.ty - expected_ty) < tolerance
    assert abs(params.tx - expected_tx) < tolerance
