import numpy as np

from biochemical_stain_assessment.registration.keypoint_register import (
    AffineParams,
    detect_and_match,
    fit_scale_translation,
    register_keypoint,
)


def _point_grid(n: int = 10) -> np.ndarray:
    rows, cols = np.meshgrid(
        np.linspace(10, 90, n), np.linspace(20, 100, n), indexing="ij"
    )
    return np.column_stack([rows.ravel(), cols.ravel()])


def _blob_image(size: int = 160) -> np.ndarray:
    rng = np.random.default_rng(7)
    img = np.zeros((size, size), dtype=np.float32)
    rr, cc = np.ogrid[:size, :size]
    for _ in range(35):
        cy = int(rng.integers(15, size - 15))
        cx = int(rng.integers(15, size - 15))
        radius = int(rng.integers(3, 8))
        img[(rr - cy) ** 2 + (cc - cx) ** 2 < radius**2] = rng.uniform(0.5, 1.0)
    return img


def test_detect_and_match_returns_arrays() -> None:
    img = _blob_image()
    src, dst = detect_and_match(img, img, n_keypoints=300)
    assert src.shape[1] == 2
    assert dst.shape[1] == 2
    assert len(src) > 0


def test_fit_scale_translation_identity() -> None:
    pts = _point_grid()
    params = fit_scale_translation(pts, pts)
    assert params is not None
    assert abs(params.scale - 1.0) < 1e-6
    assert abs(params.tx) < 1e-6
    assert abs(params.ty) < 1e-6


def test_fit_known_translation() -> None:
    dst = _point_grid()
    src = dst + np.array([10.0, 20.0])
    params = fit_scale_translation(src, dst)
    assert params is not None
    assert abs(params.scale - 1.0) < 1e-6
    assert abs(params.ty - 10.0) < 1e-6
    assert abs(params.tx - 20.0) < 1e-6


def test_fit_known_scale() -> None:
    dst = _point_grid()
    src = 1.05 * dst
    params = fit_scale_translation(src, dst)
    assert params is not None
    assert abs(params.scale - 1.05) < 1e-6
    assert abs(params.tx) < 1e-6
    assert abs(params.ty) < 1e-6


def test_fit_scale_and_translation() -> None:
    dst = _point_grid()
    src = 0.98 * dst + np.array([5.0, -8.0])
    params = fit_scale_translation(src, dst)
    assert params is not None
    assert abs(params.scale - 0.98) < 1e-6
    assert abs(params.ty - 5.0) < 1e-6
    assert abs(params.tx + 8.0) < 1e-6


def test_fit_scale_translation_returns_none_on_too_few_points() -> None:
    src = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 2.0]])
    dst = src.copy()

    assert fit_scale_translation(src, dst, min_inliers=4) is None


def test_ransac_rejects_outliers() -> None:
    rng = np.random.default_rng(3)
    dst = _point_grid()
    src = dst + np.array([6.0, -4.0])
    src[:20] = rng.uniform(0, 200, (20, 2))
    params = fit_scale_translation(src, dst)
    assert params is not None
    assert params.n_inliers < len(src)
    assert abs(params.ty - 6.0) < 0.5
    assert abs(params.tx + 4.0) < 0.5


def test_fallback_on_no_matches() -> None:
    abpas = np.full((64, 64, 3), 128, dtype=np.uint8)
    dapi = np.zeros((64, 64), dtype=np.float32)
    fallback = AffineParams(1.0, 0.0, 0.0, 0, "stage_coords", 0.0)
    _, _, params = register_keypoint(abpas, dapi, fallback_params=fallback)
    assert params == fallback


def test_register_keypoint_output_shapes() -> None:
    dapi = _blob_image(128)
    abpas = np.dstack(
        [(1.0 - dapi) * 255, (1.0 - dapi) * 220, (1.0 - dapi) * 240]
    ).astype(np.uint8)
    abpas_out, dapi_out, _ = register_keypoint(
        abpas,
        dapi,
        n_keypoints=300,
        fallback_params=AffineParams(1.0, 0.0, 0.0, 0, "stage_coords", 0.0),
    )
    assert abpas_out.shape == abpas.shape
    assert dapi_out.shape == dapi.shape


def test_register_keypoint_dapi_range() -> None:
    dapi = _blob_image(128)
    abpas = np.dstack(
        [(1.0 - dapi) * 255, (1.0 - dapi) * 220, (1.0 - dapi) * 240]
    ).astype(np.uint8)
    _, dapi_out, _ = register_keypoint(
        abpas,
        dapi,
        n_keypoints=300,
        fallback_params=AffineParams(1.0, 0.0, 0.0, 0, "stage_coords", 0.0),
    )
    assert dapi_out.min() >= 0.0
    assert dapi_out.max() <= 1.0
