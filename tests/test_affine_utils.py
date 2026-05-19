from pathlib import Path

import numpy as np
import pytest

from biochemical_stain_assessment.registration.affine_utils import (
    apply_scale_translation,
    load_affine_params,
    mutual_crop,
    save_affine_params,
)
from biochemical_stain_assessment.registration.keypoint_register import AffineParams


def test_apply_identity_transform() -> None:
    img = np.zeros((32, 32), dtype=np.float32)
    img[8:16, 8:16] = 1.0
    out = apply_scale_translation(img, 1.0, 0.0, 0.0, img.shape, order=0)
    np.testing.assert_allclose(out, img)


def test_apply_known_translation() -> None:
    img = np.zeros((40, 40), dtype=np.float32)
    img[20, 20] = 1.0
    out = apply_scale_translation(img, 1.0, 5.0, 3.0, img.shape, order=0)
    assert out[23, 25] == 1.0


def test_mutual_crop_shape() -> None:
    abpas = np.zeros((100, 120, 3), dtype=np.uint8)
    dapi = np.zeros((100, 120), dtype=np.float32)
    abpas_crop, dapi_crop = mutual_crop(abpas, dapi, margin=10)
    assert abpas_crop.shape == (80, 100, 3)
    assert dapi_crop.shape == (80, 100)


def test_mutual_crop_symmetric() -> None:
    abpas = np.arange(100 * 120 * 3, dtype=np.uint32).reshape(100, 120, 3)
    dapi = np.arange(100 * 120, dtype=np.float32).reshape(100, 120)

    abpas_crop, dapi_crop = mutual_crop(abpas, dapi, margin=10)

    np.testing.assert_array_equal(abpas_crop, abpas[10:-10, 10:-10])
    np.testing.assert_array_equal(dapi_crop, dapi[10:-10, 10:-10])


def test_mutual_crop_raises_on_mismatched_shapes() -> None:
    abpas = np.zeros((100, 120, 3), dtype=np.uint8)
    dapi = np.zeros((99, 120), dtype=np.float32)

    with pytest.raises(ValueError, match="same spatial shape"):
        mutual_crop(abpas, dapi, margin=10)


def test_save_load_affine_params(tmp_path: Path) -> None:
    params = AffineParams(
        scale=1.02, tx=3.0, ty=-2.0, n_inliers=12, method="ORB", rmse=1.5
    )
    path = tmp_path / "params.json"
    save_affine_params(params, path, "sample", "a.czi", "d.czi")
    assert load_affine_params(path) == params


def test_apply_preserves_float_range() -> None:
    img = np.zeros((32, 32), dtype=np.float32)
    img[10:20, 10:20] = 1.0
    out = apply_scale_translation(img, 1.0, 2.0, 2.0, img.shape)
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_apply_scale_translation_3channel() -> None:
    img = np.zeros((40, 40, 3), dtype=np.uint8)
    img[20, 20] = [255, 128, 64]

    out = apply_scale_translation(img, 1.0, 5.0, 3.0, img.shape[:2], order=0)

    assert out.shape == img.shape
    assert out.dtype == img.dtype
    np.testing.assert_array_equal(out[23, 25], [255, 128, 64])
