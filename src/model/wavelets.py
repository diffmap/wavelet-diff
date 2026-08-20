"""Wavelet transforms used by the forecasting pipeline.

The project stores bands in the order ``[cD1, ..., cD_level, cA_level]``.
PyWavelets stores SWT coefficients as ``[(cA_level, cD_level), ...]``.
This module keeps that conversion in one tested place.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pywt


def inverse_swt(
    coefficients: np.ndarray,
    wavelet: str,
    level: int | None = None,
) -> np.ndarray:
    """Reconstruct a signal from coefficients in the project band order.

    Args:
        coefficients: Array with shape ``(..., time, bands, features)``.
        wavelet: PyWavelets wavelet name, such as ``"db4"``.
        level: SWT level. The function infers it from the band count when omitted.

    Returns:
        Array with shape ``(..., time, features)``.

        The function zero-pads non-divisible lengths to a multiple of ``2**level``.
        It crops the reconstructed signal back to the input length.

    Raises:
        ValueError: If the input rank, band count, or level is invalid.
    """
    values = np.asarray(coefficients, dtype=np.float64)
    if values.ndim < 3:
        raise ValueError("coefficients must have shape (..., time, bands, features)")

    band_count = values.shape[-2]
    inferred_level = band_count - 1
    if inferred_level < 1:
        raise ValueError("coefficients must contain at least one detail band and one approximation band")
    if level is None:
        level = inferred_level
    if level != inferred_level:
        raise ValueError(f"expected {level + 1} bands for level={level}, got {band_count}")

    leading_shape = values.shape[:-3]
    time_length = values.shape[-3]
    transform_length = int(np.ceil(time_length / (2**level)) * (2**level))
    feature_count = values.shape[-1]
    flattened = values.reshape((-1, time_length, band_count, feature_count))
    reconstructed = np.empty((flattened.shape[0], time_length, feature_count), dtype=np.float64)

    for sample_index, sample in enumerate(flattened):
        for feature_index in range(feature_count):
            bands = sample[:, :, feature_index]
            if transform_length != time_length:
                bands = np.pad(bands, ((0, transform_length - time_length), (0, 0)))
            swt_coefficients: Sequence[tuple[np.ndarray, np.ndarray]] = [
                (bands[:, -1], bands[:, detail_level - 1])
                for detail_level in range(level, 0, -1)
            ]
            reconstructed[sample_index, :, feature_index] = pywt.iswt(
                swt_coefficients,
                wavelet,
            )[:time_length]

    return reconstructed.reshape((*leading_shape, time_length, feature_count))
