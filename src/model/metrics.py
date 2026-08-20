"""Metrics for probabilistic time-series forecasts."""

from __future__ import annotations

import numpy as np


def compute_crps(samples: np.ndarray, truth: np.ndarray) -> float:
    """Compute the ensemble CRPS averaged across forecast times."""
    forecast_samples = np.asarray(samples, dtype=np.float64)
    observed = np.asarray(truth, dtype=np.float64)
    if forecast_samples.ndim != 2 or observed.ndim != 1:
        raise ValueError("samples must have shape [n_samples, time] and truth must have shape [time]")
    if forecast_samples.shape[1] != observed.shape[0]:
        raise ValueError("samples and truth must have the same forecast length")

    term_one = np.mean(np.abs(forecast_samples - observed), axis=0)
    term_two = 0.5 * np.mean(
        np.abs(forecast_samples[:, None] - forecast_samples[None, :]),
        axis=(0, 1),
    )
    return float(np.mean(term_one - term_two))


def interval_coverage(
    samples: np.ndarray,
    truth: np.ndarray,
    central_mass: float,
) -> float:
    """Return the fraction of observations inside a central ensemble interval."""
    if not 0.0 < central_mass < 1.0:
        raise ValueError("central_mass must be between zero and one")
    forecast_samples = np.asarray(samples, dtype=np.float64)
    observed = np.asarray(truth, dtype=np.float64)
    if forecast_samples.ndim != 2 or observed.ndim != 1:
        raise ValueError("samples must have shape [n_samples, time] and truth must have shape [time]")
    lower = np.quantile(forecast_samples, (1.0 - central_mass) / 2.0, axis=0)
    upper = np.quantile(forecast_samples, 1.0 - (1.0 - central_mass) / 2.0, axis=0)
    return float(np.mean((observed >= lower) & (observed <= upper)))


def ensemble_summary(samples: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    """Return point, probabilistic, and calibration metrics for one forecast."""
    forecast_samples = np.asarray(samples, dtype=np.float64)
    observed = np.asarray(truth, dtype=np.float64)
    median = np.median(forecast_samples, axis=0)
    errors = median - observed
    return {
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "crps": compute_crps(forecast_samples, observed),
        "coverage_50": interval_coverage(forecast_samples, observed, 0.50),
        "coverage_90": interval_coverage(forecast_samples, observed, 0.90),
    }
