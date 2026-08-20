# src/model/config.py

import os
import random

import numpy as np
import torch

PROJECT_ROOT      = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_ROOT         = os.path.join(PROJECT_ROOT, "data")
WAVELETS_TRAIN_DIR = os.path.join(DATA_ROOT, "wavelets", "train wavelet")
WAVELETS_TEST_DIR  = os.path.join(DATA_ROOT, "wavelets", "test wavelet")
CHECKPOINT_DIR     = os.path.join(PROJECT_ROOT, "outputs", "checkpoints")
PREDICTION_OUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "predictions")


def ensure_output_dirs() -> None:
    """Create the local output directories this run needs.

    Call this explicitly from entry points (trainer, predictandsave). It must
    not run on import, so importing `config` stays a side-effect-free operation.
    """
    for path in (WAVELETS_TRAIN_DIR, WAVELETS_TEST_DIR, CHECKPOINT_DIR, PREDICTION_OUT_DIR):
        os.makedirs(path, exist_ok=True)


CONFIG = {
    "project_root": PROJECT_ROOT,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "history_len": 50,
    "predict_len": 20,
    "wavelet": "db4",
    "wavelet_level": 4,  # 4 levels → 5 bands

    # Paths
    "train_data_path": os.path.join(DATA_ROOT, "wavelets", "train wavelet"),
    "test_data_path":  os.path.join(DATA_ROOT, "wavelets", "test wavelet"),
    "wavelets_path":  os.path.join(DATA_ROOT, "wavelets", "train wavelet"),
    "checkpoint_dir": CHECKPOINT_DIR,
    # Default checkpoint for standalone inference (predictandsave.py __main__).
    # Overridden by find_latest_checkpoint() when no checkpoint of this name exists.
    "checkpoint_path": os.path.join(CHECKPOINT_DIR, "score_transformer_ep300.pth"),
    "prediction_output_dir": PREDICTION_OUT_DIR,
    "test_returns_path": os.path.join(DATA_ROOT, "Testing Data", "bitcoin_2010-07-29_2025-04-25_test.csv"),

    # Training hyperparams
    "model_dim": 256,
    "num_heads": 8,
    "num_layers": 4,
    "mlp_ratio": 4.0,
    "drop_rate": 0.1,
    "attn_drop_rate": 0.1,
    "samples_per_epoch": 400,
    "batch_size": 32,
    "pin_memory": True,
    "num_workers": 4,
    "prefetch_factor": 2,
    "grad_accum_steps": 1,
    "n_epochs": 500,
    "lr": 1e-4,
    "ema_decay": 0.999,
    "checkpoint_freq": 10,
    "save_name": "score_transformer",
    "mask_ratio": 0.1,
    "cond_drop_prob": 0.2,

    # Diffusion / sampling
    "num_diffusion_timesteps": 500,
    "classifier_free_guidance_weight": .25,
    "regular_samples": 1,
    "high_samples": 200,
    "num_regular_windows": 1,
    "include_high_sample": True,
    "input_dim": 1,

    # Normalization files (still from train‐wavelet)
    #   train_wavelet/wavelet_means.pt  (shape [5,feat_dim])
    #   train_wavelet/wavelet_stds.pt   (shape [5,feat_dim])
}


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_latest_checkpoint(save_name: str = CONFIG["save_name"], checkpoint_dir: str = CHECKPOINT_DIR) -> str:
    """Return the highest-epoch checkpoint for `save_name` in `checkpoint_dir`.

    Falls back to CONFIG["checkpoint_path"] if no matching checkpoint exists yet.
    """
    prefix, suffix = f"{save_name}_ep", ".pth"
    candidates = []
    if os.path.isdir(checkpoint_dir):
        for fname in os.listdir(checkpoint_dir):
            if fname.startswith(prefix) and fname.endswith(suffix):
                epoch_str = fname[len(prefix):-len(suffix)]
                if epoch_str.isdigit():
                    candidates.append((int(epoch_str), fname))

    if not candidates:
        return CONFIG["checkpoint_path"]

    _, latest_fname = max(candidates)
    return os.path.join(checkpoint_dir, latest_fname)
