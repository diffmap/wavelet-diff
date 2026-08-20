"""Evaluate a compatible Wavelet Diff checkpoint at rolling test origins."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.model.config import CONFIG, seed_everything
from src.model.data import WaveletSlidingWindowDataset, load_folder_as_tensor
from src.model.metrics import ensemble_summary
from src.model.model import ScoreTransformerNet
from src.model.predictandsave import (
    load_norm_factors,
    reverse_sde_sampler,
    unnormalize_wavelet,
)
from src.model.sde import VPSDE
from src.model.wavelets import inverse_swt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--max-origins", type=int, default=None)
    parser.add_argument("--diffusion-steps", type=int, default=None)
    parser.add_argument("--guidance-weight", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default=None)
    return parser.parse_args()


def _build_model(checkpoint: dict, device: torch.device) -> tuple[ScoreTransformerNet, dict]:
    """Build and validate the current model against checkpoint metadata."""
    checkpoint_cfg = checkpoint.get("cfg", {})
    level = int(checkpoint_cfg.get("wavelet_level", CONFIG["wavelet_level"]))
    model = ScoreTransformerNet(
        input_dim=int(checkpoint_cfg.get("input_dim", CONFIG["input_dim"])),
        history_len=int(checkpoint_cfg.get("history_len", CONFIG["history_len"])),
        predict_len=int(checkpoint_cfg.get("predict_len", CONFIG["predict_len"])),
        model_dim=int(checkpoint_cfg.get("model_dim", CONFIG["model_dim"])),
        num_heads=int(checkpoint_cfg.get("num_heads", CONFIG["num_heads"])),
        num_layers=int(checkpoint_cfg.get("num_layers", CONFIG["num_layers"])),
        wavelet_levels=level,
        mlp_ratio=float(checkpoint_cfg.get("mlp_ratio", CONFIG["mlp_ratio"])),
        drop_rate=float(checkpoint_cfg.get("drop_rate", CONFIG["drop_rate"])),
        attn_drop_rate=float(checkpoint_cfg.get("attn_drop_rate", CONFIG["attn_drop_rate"])),
    ).to(device)

    state = checkpoint.get("ema", checkpoint.get("model", checkpoint))
    current_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state.items()
        if key in current_state and current_state[key].shape == value.shape
    }
    coverage = len(compatible) / len(current_state)
    if coverage < 0.9:
        raise ValueError(
            f"Checkpoint is incompatible with the current model: matched "
            f"{len(compatible)}/{len(current_state)} tensors ({coverage:.1%}). "
            "Retrain the model with the current source code."
        )
    current_state.update(compatible)
    model.load_state_dict(current_state)
    model.eval()
    return model, checkpoint_cfg


def _reconstruct_returns(
    history_norm: torch.Tensor,
    future_norm: torch.Tensor,
    means: np.ndarray,
    stds: np.ndarray,
    wavelet: str,
    history_len: int,
) -> np.ndarray:
    """Unnormalize and reconstruct sampled future returns from SWT bands."""
    history = unnormalize_wavelet(history_norm.cpu().numpy(), means, stds)
    future = unnormalize_wavelet(future_norm.cpu().numpy(), means, stds)
    full_wavelet = np.concatenate([history, future], axis=1)
    reconstructed = inverse_swt(full_wavelet, wavelet)
    return reconstructed[:, history_len:, :].squeeze(-1)


def evaluate_checkpoint(
    checkpoint_path: Path,
    output_path: Path,
    samples: int,
    stride: int,
    max_origins: int | None,
    diffusion_steps: int | None,
    seed: int,
    device_name: str | None,
    guidance_weight: float | None = None,
) -> pd.DataFrame:
    """Run rolling-origin checkpoint evaluation and save aggregate metrics."""
    if samples < 1 or stride < 1:
        raise ValueError("samples and stride must be positive")
    seed_everything(seed)
    device = torch.device(device_name or CONFIG["device"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model, checkpoint_cfg = _build_model(checkpoint, device)

    history_len = model.history_len
    predict_len = model.predict_len
    level = int(checkpoint_cfg.get("wavelet_level", CONFIG["wavelet_level"]))
    wavelet = checkpoint_cfg.get("wavelet", CONFIG["wavelet"])
    steps = diffusion_steps or int(checkpoint_cfg.get("num_diffusion_timesteps", CONFIG["num_diffusion_timesteps"]))
    guidance = float(
        checkpoint_cfg.get(
            "classifier_free_guidance_weight",
            CONFIG["classifier_free_guidance_weight"],
        )
    )
    if guidance_weight is not None:
        guidance = guidance_weight

    test_tensor = load_folder_as_tensor(
        root_folder=None,
        precompute_wavelets_path=CONFIG["test_data_path"],
        wavelet=wavelet,
        level=level,
    )
    dataset = WaveletSlidingWindowDataset(test_tensor, history_len, predict_len)
    returns_path = Path(CONFIG["test_returns_path"])
    returns = pd.read_csv(returns_path)["Close"].astype(float).to_numpy()
    norm_folder = Path(CONFIG["wavelets_path"])
    means, stds = load_norm_factors(str(norm_folder), level)
    sde_model = VPSDE(
        bmin=float(checkpoint_cfg.get("sde_bmin", 0.1)),
        bmax=float(checkpoint_cfg.get("sde_bmax", 20.0)),
    )

    window_indices = list(range(0, len(dataset), stride))
    if max_origins is not None:
        window_indices = window_indices[:max_origins]
    rows = []
    for window_index in window_indices:
        history_norm, _ = dataset[window_index]
        history_batch = history_norm.unsqueeze(0).expand(samples, -1, -1, -1).to(device)
        future_norm = reverse_sde_sampler(
            model_net=model,
            sde_model=sde_model,
            history=history_batch,
            num_steps=steps,
            guidance_weight=guidance,
            device=device,
        )
        predicted_returns = _reconstruct_returns(
            history_batch,
            future_norm,
            means,
            stds,
            wavelet,
            history_len,
        )
        truth_start = window_index + history_len
        truth = returns[truth_start : truth_start + predict_len]
        if truth.shape[0] != predict_len:
            raise ValueError(f"test return window {window_index} has length {truth.shape[0]}")
        rows.append(
            {
                "window_index": window_index,
                "origin": truth_start,
                **ensemble_summary(predicted_returns, truth),
            }
        )

    per_origin = pd.DataFrame(rows)
    aggregate = pd.DataFrame(
        [{"model": "wavelet_diffusion", "n_origins": len(per_origin), **{
            name: float(per_origin[name].mean())
            for name in ("mae", "rmse", "crps", "coverage_50", "coverage_90")
        }}]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    aggregate.to_csv(output_path, index=False)
    per_origin.to_csv(output_path.with_name(f"{output_path.stem}_per_origin.csv"), index=False)
    return aggregate


def main() -> None:
    args = parse_args()
    metrics = evaluate_checkpoint(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        samples=args.samples,
        stride=args.stride,
        max_origins=args.max_origins,
        diffusion_steps=args.diffusion_steps,
        seed=args.seed,
        device_name=args.device,
        guidance_weight=args.guidance_weight,
    )
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
