"""Small forecasting baselines for rolling-origin comparisons."""

from __future__ import annotations

import numpy as np


def zero_return(history: np.ndarray, horizon: int, samples: int = 1) -> np.ndarray:
    """Forecast zero return at every future time."""
    del history
    return np.zeros((samples, horizon), dtype=np.float64)


def last_return(history: np.ndarray, horizon: int, samples: int = 1) -> np.ndarray:
    """Repeat the latest observed return at every future time."""
    values = np.asarray(history, dtype=np.float64)
    if values.size == 0:
        raise ValueError("history must contain at least one value")
    return np.full((samples, horizon), values[-1], dtype=np.float64)


def historical_bootstrap(
    history: np.ndarray,
    horizon: int,
    samples: int = 100,
    seed: int = 0,
) -> np.ndarray:
    """Sample independent future returns from the observed history."""
    values = np.asarray(history, dtype=np.float64)
    if values.size == 0:
        raise ValueError("history must contain at least one value")
    rng = np.random.default_rng(seed)
    return rng.choice(values, size=(samples, horizon), replace=True)
