from pathlib import Path

import numpy as np
import pytest

from biochemical_stain_assessment.registration.normalise_abpas import (
    DEFAULT_STAIN_MATRIX,
    estimate_stain_matrix,
    load_stain_matrix,
    normalise_abpas,
    save_stain_matrix,
)


@pytest.fixture
def synthetic_abpas() -> np.ndarray:
    rng = np.random.default_rng(42)
    h, w = 128, 128
    base = np.array([205, 155, 185], dtype=np.float32)
    img = np.clip(base + rng.normal(0, 25, (h, w, 3)), 0, 255).astype(np.uint8)
    img[32:96, 40:88] = np.clip(img[32:96, 40:88] - [60, 35, 45], 0, 255)
    return img


def test_output_dtype_shape_and_range(synthetic_abpas: np.ndarray) -> None:
    out = normalise_abpas(synthetic_abpas)
    assert out.shape == synthetic_abpas.shape
    assert out.dtype == np.uint8


def test_output_range() -> None:
    rng = np.random.default_rng(10)
    rgb = rng.integers(40, 240, (96, 96, 3), dtype=np.uint8)
    rgb[24:72, 28:68] = np.clip(rgb[24:72, 28:68] - [45, 30, 35], 0, 255)
    out = normalise_abpas(rgb)
    assert out.min() >= 0
    assert out.max() <= 255


def test_identity_on_reference() -> None:
    rng = np.random.default_rng(4)
    concentrations = rng.uniform(0.0, 0.8, (96, 96, 2))
    od = concentrations @ DEFAULT_STAIN_MATRIX
    rgb = np.clip(255.0 * np.exp(-od), 0, 255).astype(np.uint8)

    stain_matrix = estimate_stain_matrix(rgb, percentile=100.0)
    out = normalise_abpas(rgb, stain_matrix_target=stain_matrix, percentile=100.0)

    np.testing.assert_allclose(out, rgb, atol=5)


def test_estimate_stain_matrix_shape(synthetic_abpas: np.ndarray) -> None:
    matrix = estimate_stain_matrix(synthetic_abpas)
    assert matrix.shape == (2, 3)
    np.testing.assert_allclose(np.linalg.norm(matrix, axis=1), 1.0)


def test_save_load_stain_matrix(tmp_path: Path, synthetic_abpas: np.ndarray) -> None:
    matrix = estimate_stain_matrix(synthetic_abpas)
    path = tmp_path / "stain.json"
    save_stain_matrix(matrix, path, sample_id="synthetic", estimated_from="test")
    np.testing.assert_allclose(load_stain_matrix(path), matrix, atol=1e-10)


def test_raises_on_blank_image() -> None:
    blank = np.full((64, 64, 3), 255, dtype=np.uint8)
    with pytest.raises(ValueError):
        estimate_stain_matrix(blank)


def test_handles_zero_pixels(synthetic_abpas: np.ndarray) -> None:
    synthetic_abpas = synthetic_abpas.copy()
    synthetic_abpas[:4, :4] = 0
    out = normalise_abpas(synthetic_abpas)
    assert np.isfinite(out).all()
