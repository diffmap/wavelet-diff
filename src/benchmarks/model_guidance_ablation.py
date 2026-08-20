"""Measure checkpoint sensitivity to classifier-free guidance weight."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.benchmarks.model_rolling_evaluation import evaluate_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--guidance-weights", type=float, nargs="+", default=[0.0, 0.25, 1.0])
    parser.add_argument("--samples", type=int, default=20)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--diffusion-steps", type=int, default=10)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = []
    for weight in args.guidance_weights:
        metrics = evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            output_path=args.output.with_name(f"{args.output.stem}_guidance_{weight:g}.csv"),
            samples=args.samples,
            stride=args.stride,
            max_origins=None,
            diffusion_steps=args.diffusion_steps,
            seed=args.seed,
            device_name=args.device,
            guidance_weight=weight,
        )
        row = metrics.iloc[0].to_dict()
        row["guidance_weight"] = weight
        rows.append(row)
    results = pd.DataFrame(rows)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output, index=False)
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
