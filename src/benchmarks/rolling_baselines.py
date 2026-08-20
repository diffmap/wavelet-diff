"""Run leakage-safe rolling-origin baseline evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.model.baselines import historical_bootstrap, last_return, zero_return
from src.model.evaluation import rolling_origin_evaluate


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="CSV file with a Close column.")
    parser.add_argument("--output", type=Path, required=True, help="Output CSV for aggregate metrics.")
    parser.add_argument("--history-len", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_log_returns(path: Path) -> np.ndarray:
    """Load sorted close prices and return their chronological log returns."""
    frame = pd.read_csv(path)
    if "Close" not in frame:
        raise ValueError(f"{path} must contain a Close column")
    close = frame["Close"].astype(float).to_numpy()
    if close.size < 2 or np.any(close <= 0.0):
        raise ValueError("Close prices must contain at least two positive values")
    return np.diff(np.log(close))


def run_baselines(
    series: np.ndarray,
    history_len: int,
    horizon: int,
    stride: int,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    """Evaluate all built-in baselines and return one row per baseline."""
    factories = {
        "zero_return": lambda history: zero_return(history, horizon, samples),
        "last_return": lambda history: last_return(history, horizon, samples),
        "historical_bootstrap": lambda history: historical_bootstrap(
            history, horizon, samples, seed
        ),
    }
    rows = []
    for name, forecast_fn in factories.items():
        _, summary = rolling_origin_evaluate(
            series,
            forecast_fn,
            history_len=history_len,
            horizon=horizon,
            stride=stride,
        )
        rows.append({"model": name, **summary})
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    metrics = run_baselines(
        load_log_returns(args.input),
        history_len=args.history_len,
        horizon=args.horizon,
        stride=args.stride,
        samples=args.samples,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(args.output, index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
