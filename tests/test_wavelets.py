import numpy as np
import pywt
import pytest

from src.model.wavelets import inverse_swt


def _project_band_order(signal: np.ndarray, wavelet: str, level: int) -> np.ndarray:
    coefficients = pywt.swt(signal, wavelet, level=level)
    details = [pair[1] for pair in reversed(coefficients)]
    approximation = coefficients[0][0]
    return np.stack([*details, approximation], axis=-1)[..., np.newaxis]


def test_inverse_swt_reconstructs_project_band_order() -> None:
    signal = np.sin(np.linspace(0.0, 8.0 * np.pi, 64))
    coefficients = _project_band_order(signal, "db2", level=2)

    reconstructed = inverse_swt(coefficients, "db2")

    np.testing.assert_allclose(reconstructed[:, 0], signal, atol=1e-6)


def test_inverse_swt_supports_batch_and_feature_dimensions() -> None:
    signals = np.stack(
        [
            np.sin(np.linspace(0.0, 4.0 * np.pi, 64)),
            np.cos(np.linspace(0.0, 4.0 * np.pi, 64)),
        ],
        axis=-1,
    )
    per_feature = [
        _project_band_order(signals[:, feature], "db2", level=2)[..., 0]
        for feature in range(signals.shape[-1])
    ]
    coefficients = np.stack(per_feature, axis=-1)[np.newaxis, ...]

    reconstructed = inverse_swt(coefficients, "db2")

    np.testing.assert_allclose(reconstructed[0], signals, atol=1e-6)


def test_inverse_swt_pads_non_divisible_time_length() -> None:
    coefficients = np.zeros((1, 20, 5, 1), dtype=np.float64)

    reconstructed = inverse_swt(coefficients, "db4", level=4)

    assert reconstructed.shape == (1, 20, 1)
    assert np.isfinite(reconstructed).all()


def test_inverse_swt_rejects_wrong_level() -> None:
    coefficients = np.zeros((32, 2, 1), dtype=np.float64)

    with pytest.raises(ValueError, match="expected 3 bands"):
        inverse_swt(coefficients, "db2", level=2)
