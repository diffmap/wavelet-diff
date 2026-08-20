"""Measure baseline sensitivity to history length and forecast horizon."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.benchmarks.rolling_baselines import load_log_returns, run_baselines


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-lens", type=int, nargs="+", default=[20, 50, 100])
    parser.add_argument("--horizons", type=int, nargs="+", default=[5, 20])
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def run_ablation(
    series,
    history_lens: list[int],
    horizons: list[int],
    stride: int,
    samples: int,
    seed: int,
) -> pd.DataFrame:
    """Evaluate each baseline across the requested protocol settings."""
    results = []
    for history_len in history_lens:
        for horizon in horizons:
            metrics = run_baselines(series, history_len, horizon, stride, samples, seed)
            metrics.insert(0, "horizon", horizon)
            metrics.insert(0, "history_len", history_len)
            results.append(metrics)
    return pd.concat(results, ignore_index=True)


def main() -> None:
    args = parse_args()
    results = run_ablation(
        load_log_returns(args.input),
        args.history_lens,
        args.horizons,
        args.stride,
        args.samples,
        args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
