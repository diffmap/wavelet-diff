"""Leakage-safe rolling-origin evaluation for forecast functions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.model.metrics import ensemble_summary


@dataclass(frozen=True)
class ForecastRecord:
    """One forecast origin and its held-out future."""

    origin: int
    samples: np.ndarray
    truth: np.ndarray


def rolling_origin_evaluate(
    series: np.ndarray,
    forecast_fn: Callable[[np.ndarray], np.ndarray],
    history_len: int,
    horizon: int,
    start: int | None = None,
    stop: int | None = None,
    stride: int = 1,
) -> tuple[list[ForecastRecord], dict[str, float]]:
    """Evaluate a forecast function at multiple chronological origins.

    The function passes only observations before each origin to ``forecast_fn``.
    The returned forecast must have shape ``[n_samples, horizon]``.
    """
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("series must have shape [time]")
    if history_len < 1 or horizon < 1 or stride < 1:
        raise ValueError("history_len, horizon, and stride must be positive")

    first_origin = history_len if start is None else max(start, history_len)
    last_origin = values.shape[0] - horizon if stop is None else min(stop, values.shape[0] - horizon)
    if first_origin > last_origin:
        raise ValueError("the series does not contain a complete evaluation window")

    records: list[ForecastRecord] = []
    summaries: list[dict[str, float]] = []
    for origin in range(first_origin, last_origin + 1, stride):
        history = values[origin - history_len : origin].copy()
        truth = values[origin : origin + horizon].copy()
        samples = np.asarray(forecast_fn(history), dtype=np.float64)
        if samples.ndim == 1:
            samples = samples[np.newaxis, :]
        if samples.ndim != 2 or samples.shape[1] != horizon:
            raise ValueError(f"forecast_fn must return [n_samples, {horizon}], got {samples.shape}")
        records.append(ForecastRecord(origin=origin, samples=samples, truth=truth))
        summaries.append(ensemble_summary(samples, truth))

    aggregate = {name: float(np.mean([summary[name] for summary in summaries])) for name in summaries[0]}
    aggregate["n_origins"] = float(len(records))
    return records, aggregate
