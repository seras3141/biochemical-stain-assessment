import numpy as np
import pytest

from biochemical_stain_assessment.registration.enhance_dapi import (
    clip_percentile,
    enhance_dapi,
)


@pytest.fixture
def synthetic_dapi() -> np.ndarray:
    rng = np.random.default_rng(0)
    h, w = 128, 128
    img = rng.integers(100, 800, (h, w), dtype=np.uint16)
    rr, cc = np.ogrid[:h, :w]
    for _ in range(20):
        cy = int(rng.integers(15, h - 15))
        cx = int(rng.integers(15, w - 15))
        radius = int(rng.integers(4, 10))
        mask = (rr - cy) ** 2 + (cc - cx) ** 2 < radius**2
        img[mask] = rng.integers(3000, 5000)
    return img


def test_output_dtype_shape_and_range(synthetic_dapi: np.ndarray) -> None:
    out = enhance_dapi(synthetic_dapi)
    assert out.shape == synthetic_dapi.shape
    assert out.dtype == np.float32
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_all_zeros_input() -> None:
    out = enhance_dapi(np.zeros((32, 32), dtype=np.uint16))
    assert out.dtype == np.float32
    assert np.count_nonzero(out) == 0


def test_clip_percentile_clips_hot_pixel() -> None:
    img = np.ones((32, 32), dtype=np.float32)
    img[0, 0] = 1000.0
    out = clip_percentile(img, low=0, high=99)
    assert out[0, 0] <= 1.0
    assert np.isfinite(out).all()


def test_clahe_increases_contrast() -> None:
    img = np.full((128, 128), 1000, dtype=np.uint16)
    img[40:88, 40:88] = 1200
    before = clip_percentile(img).std()
    after = enhance_dapi(img).std()
    assert after > 0
    assert after >= before * 0.95


def test_tophat_enabled(synthetic_dapi: np.ndarray) -> None:
    out = enhance_dapi(synthetic_dapi, tophat=True, tophat_radius=5)
    assert out.shape == synthetic_dapi.shape
    assert out.min() >= 0.0


def test_clahe_kernel_size_none(
    synthetic_dapi: np.ndarray, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(
        "DEBUG", logger="biochemical_stain_assessment.registration.enhance_dapi"
    )

    out = enhance_dapi(synthetic_dapi, clahe_kernel_size=None)

    assert out.shape == synthetic_dapi.shape
    assert "DAPI CLAHE kernel size: 16" in caplog.text
