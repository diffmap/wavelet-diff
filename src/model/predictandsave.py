# src/model/predictandsave.py

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from datetime import datetime

from src.model.model import ScoreTransformerNet
from src.model.sde import VPSDE
from src.model.config import CONFIG, find_latest_checkpoint, ensure_output_dirs
from src.model.data import load_folder_as_tensor, WaveletSlidingWindowDataset


def compute_crps(samples: np.ndarray, truth: np.ndarray) -> float:
    """
    Continuous Ranked Probability Score for one-dimensional output.
    `samples`: shape [N_paths, T], `truth`: shape [T].
    """
    term1 = np.mean(np.abs(samples - truth), axis=0)
    term2 = 0.5 * np.mean(
        np.abs(samples[:, None] - samples[None, :]), axis=(0, 1)
    )
    return float(np.mean(term1 - term2))


def reverse_sde_sampler(
    model_net: ScoreTransformerNet,
    sde_model: VPSDE,
    history: torch.Tensor,
    num_steps: int,
    guidance_weight: float,
    device: torch.device
) -> torch.Tensor:
    """
    Reverse‐SDE sampler (Euler–Maruyama) operating in normalized wavelet space.

    Args:
      - model_net:         Trained ScoreTransformerNet (eval mode).
      - sde_model:         Instance of VPSDE for drift & diffusion.
      - history:           Tensor [B, history_len, L+1, feat_dim] of normalized history.
      - num_steps:         Number of diffusion timesteps (e.g. 500).
      - guidance_weight:   Classifier‐free guidance weight (e.g. 3.0).
      - device:            torch.device.

    Returns:
      - x_norm: [B, predict_len, L+1, feat_dim] normalized wavelet‐latent predictions.
    """
    model_net.eval()
    B, H, Lp, D = history.shape
    predict_len = CONFIG["predict_len"]

    # Initialize x_T ∼ N(0, I) in normalized wavelet space
    x = torch.randn((B, predict_len, Lp, D), device=device)
    dt = -1.0 / num_steps

    with torch.no_grad():
        for i in tqdm(range(num_steps - 1, -1, -1), desc="Reverse SDE Sampling", ncols=100):
            t_val = max(i / num_steps, 1e-5)
            t = torch.full((B, 1), t_val, device=device)

            # Classifier‐free guidance steps
            score_cond = model_net(x, history, t, cond_drop_prob=0.0)   # [B, T, L+1, D]
            score_uncond = model_net(x, history, t, cond_drop_prob=1.0) # [B, T, L+1, D]
            score = (1.0 + guidance_weight) * score_cond - guidance_weight * score_uncond

            # Compute drift & diffusion
            drift = sde_model.f(x, t)     # [B, T, L+1, D]
            diffusion = sde_model.g(t)    # [B, 1]

            score_term = -0.5 * (diffusion ** 2).view(B, 1, 1, 1) * score
            noise = torch.randn_like(x)
            x = x + (drift + score_term) * dt + diffusion.view(B, 1, 1, 1) * ((-dt) ** 0.5) * noise

    return x  # [B, predict_len, L+1, D]


def load_norm_factors(
    norm_folder: str,
    level: int
) -> (np.ndarray, np.ndarray):
    """
    Load wavelet‐level normalization factors saved during training.

    Expects in `norm_folder`:
      - wavelet_means.pt → tensor [L+1, 1]
      - wavelet_stds.pt  → tensor [L+1, 1]

    Band order: [cD1, cD2, cD3, cD4, cA4] (detail → approx)
    """
    band_count = level + 1
    means_f = os.path.join(norm_folder, "wavelet_means.pt")
    stds_f = os.path.join(norm_folder, "wavelet_stds.pt")

    if not os.path.isfile(means_f) or not os.path.isfile(stds_f):
        raise FileNotFoundError(f"Normalization files missing in {norm_folder}")

    wavelet_means = torch.load(means_f)  # [L+1, 1]
    wavelet_stds = torch.load(stds_f)    # [L+1, 1]

    wm = wavelet_means.numpy()
    ws = wavelet_stds.numpy()

    if wm.shape[0] != band_count or ws.shape[0] != band_count:
        raise ValueError(
            f"Expected normalization shape[0]={band_count}, got means {wm.shape}, stds {ws.shape}"
        )
    
    # Ensure we have [L+1, 1] shape
    if len(wm.shape) != 2 or wm.shape[1] != 1:
        raise ValueError(f"Expected means shape [L+1, 1], got {wm.shape}")
    if len(ws.shape) != 2 or ws.shape[1] != 1:
        raise ValueError(f"Expected stds shape [L+1, 1], got {ws.shape}")
        
    return wm, ws


def unnormalize_wavelet(
    x_norm: np.ndarray,
    wavelet_means: np.ndarray,
    wavelet_stds: np.ndarray
) -> np.ndarray:
    """
    Unnormalize each wavelet band: x = x_norm * std + mean.

    Args:
      - x_norm:         [B, T, L+1, feat_dim] normalized coefficients.
      - wavelet_means:  [L+1, 1] mean values.
      - wavelet_stds:   [L+1, 1] std values.

    Band order: [cD1, cD2, cD3, cD4, cA4] (detail → approx)
    Returns:
      - x_unnorm: [B, T, L+1, feat_dim] unnormalized coefficients.
    """
    # No need to change broadcasting, NumPy will handle [L+1,1] correctly
    return x_norm * wavelet_stds[None, None, :, :] + wavelet_means[None, None, :, :]


def returns_to_price(returns: np.ndarray, start_price: float = 100.0) -> np.ndarray:
    """
    Convert a 1D array of returns into a price series, starting from `start_price`.
    """
    cum_returns = np.cumsum(returns)
    return start_price * np.exp(cum_returns, dtype=np.float64)


def plot_wavelet_comparison(x_norm, x_unnorm, fut_norm, fut_unnorm, window_dir, band_names=None):
    """Plot temporal evolution of predicted vs actual wavelet coefficients for all bands."""
    if band_names is None:
        band_names = ['cD1', 'cD2', 'cD3', 'cD4', 'cA4']
    
    B, T, L, D = x_unnorm.shape
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Create two large figures - one for normalized, one for unnormalized
    for plot_type in ['normalized', 'unnormalized']:
        # Create figure with L subplots stacked vertically
        plt.figure(figsize=(15, 4*L))
        
        # Select data based on plot type
        if plot_type == 'normalized':
            data = x_norm
            true_data = fut_norm[0]
            ylabel_base = 'Normalized Coefficient Value'
            title = 'Normalized Wavelet Coefficients'
        else:
            data = x_unnorm
            true_data = fut_unnorm[0]
            ylabel_base = 'Unnormalized Coefficient Value'
            title = 'Unnormalized Wavelet Coefficients'
        
        # Plot each band in its own subplot
        for band_idx in range(L):
            plt.subplot(L, 1, band_idx + 1)
            
            # Get predicted coefficients for this band over time
            pred_coeffs = data[:, :, band_idx, 0]  # Shape: [B, T]
            true_coeffs = true_data[:, band_idx, 0]  # Shape: [T]
            
            # Calculate statistics
            median_pred = np.median(pred_coeffs, axis=0)
            q25 = np.percentile(pred_coeffs, 25, axis=0)
            q75 = np.percentile(pred_coeffs, 75, axis=0)
            
            # Plot uncertainty band
            plt.fill_between(range(T), q25, q75, color=colors[band_idx], alpha=0.2)
            
            # Plot median prediction and actual values
            plt.plot(range(T), median_pred, '--', color=colors[band_idx], 
                    label='Predicted', linewidth=2)
            plt.plot(range(T), true_coeffs, '-', color=colors[band_idx], 
                    label='Actual', linewidth=2)
            
            # Add band-specific statistics
            stats_text = f"Range: [{true_coeffs.min():.4f}, {true_coeffs.max():.4f}]\n"
            stats_text += f"MAE: {np.mean(np.abs(median_pred - true_coeffs)):.4f}"
            plt.text(0.02, 0.95, stats_text, transform=plt.gca().transAxes, 
                    verticalalignment='top', fontsize=10,
                    bbox=dict(facecolor='white', alpha=0.8))
            
            plt.xlabel('Time Step')
            plt.ylabel(f'{ylabel_base}\n{band_names[band_idx]}')
            plt.title(f'Wavelet Band: {band_names[band_idx]}')
            plt.grid(True, alpha=0.3)
            plt.legend()
        
        plt.suptitle(title, y=1.02, fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(window_dir, f'wavelet_bands_{plot_type}.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Print statistics for each band
        print(f"\n{plot_type.capitalize()} Statistics:")
        for band_idx in range(L):
            pred_coeffs = data[:, :, band_idx, 0]
            true_coeffs = true_data[:, band_idx, 0]
            median_pred = np.median(pred_coeffs, axis=0)
            
            print(f"\n{band_names[band_idx]}:")
            print(f"  Actual range: [{true_coeffs.min():.4f}, {true_coeffs.max():.4f}]")
            print(f"  Predicted range: [{median_pred.min():.4f}, {median_pred.max():.4f}]")
            print(f"  Mean absolute error: {np.mean(np.abs(median_pred - true_coeffs)):.4f}")


def plot_price_comparison(history_returns, future_returns, pred_returns, window_dir):
    """Plot price comparison between predicted and actual prices."""
    
    def returns_to_price(returns, start_price=100.0):
        """Convert returns to price series safely."""
        log_prices = np.log(start_price) + np.cumsum(returns)
        return np.exp(log_prices, dtype=np.float64)
    
    # Scale predicted returns to match historical volatility
    hist_std = np.std(history_returns)
    hist_mean = np.mean(history_returns)
    pred_std = np.std(pred_returns)
    pred_mean = np.mean(pred_returns)
    pred_returns_scaled = (pred_returns - pred_mean) * (hist_std / pred_std) + hist_mean
    
    # Convert to prices
    history_price = returns_to_price(history_returns)
    true_future_price = returns_to_price(future_returns, start_price=history_price[-1])
    
    # Convert predicted returns to prices
    pred_prices = []
    for i in range(pred_returns_scaled.shape[0]):
        try:
            prices = returns_to_price(pred_returns_scaled[i], start_price=history_price[-1])
            pred_prices.append(prices)
        except Exception:
            continue
    pred_prices = np.array(pred_prices)
    
    if len(pred_prices) == 0:
        print("Warning: Could not convert any predicted returns to prices!")
        return
    
    # Calculate statistics
    median_pred = np.median(pred_prices, axis=0)
    q25 = np.percentile(pred_prices, 25, axis=0)
    q75 = np.percentile(pred_prices, 75, axis=0)
    q10 = np.percentile(pred_prices, 10, axis=0)
    q90 = np.percentile(pred_prices, 90, axis=0)
    
    # Plot
    plt.figure(figsize=(12, 6))
    plt.plot(
        np.arange(-len(history_returns), 0), history_price,
        color='blue', label='History', linewidth=2
    )
    plt.plot(
        np.arange(len(future_returns)), true_future_price,
        color='green', label='True Future', linewidth=2
    )
    plt.plot(
        np.arange(len(future_returns)), median_pred,
        linestyle='--', color='red', label='Median Prediction', linewidth=2
    )
    plt.fill_between(
        np.arange(len(future_returns)), q10, q90,
        color='red', alpha=0.1, label='80% Interval'
    )
    plt.fill_between(
        np.arange(len(future_returns)), q25, q75,
        color='red', alpha=0.2, label='IQR (25%–75%)'
    )
    
    plt.title("Price Prediction vs Actual")
    plt.xlabel("Time Step")
    plt.ylabel("Price")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(window_dir, "price_comparison.png"), dpi=300)
    plt.close()
    
    # Save statistics
    stats = {
        'start_price': float(history_price[-1]),
        'true_end_price': float(true_future_price[-1]),
        'true_min_price': float(true_future_price.min()),
        'true_max_price': float(true_future_price.max()),
        'pred_median': float(median_pred[-1]),
        'pred_25th': float(q25[-1]),
        'pred_75th': float(q75[-1]),
        'pred_10th': float(q10[-1]),
        'pred_90th': float(q90[-1]),
    }
    
    # Print statistics
    print("\nPrice Statistics:")
    print(f"Starting Price: ${stats['start_price']:.2f}")
    print("\nTrue Future Prices:")
    print(f"End Price: ${stats['true_end_price']:.2f}")
    print(f"Range: ${stats['true_min_price']:.2f} to ${stats['true_max_price']:.2f}")
    print(f"\nPredicted Prices (at t={len(future_returns)}):")
    print(f"Median: ${stats['pred_median']:.2f}")
    print(f"25th-75th: ${stats['pred_25th']:.2f} to ${stats['pred_75th']:.2f}")
    print(f"10th-90th: ${stats['pred_10th']:.2f} to ${stats['pred_90th']:.2f}")
    
    return stats


def create_windows(returns, window_size, step_size=1):
    """Create overlapping windows from a sequence of returns.
    
    Args:
        returns: numpy array of returns
        window_size: size of each window
        step_size: number of steps between windows
        
    Returns:
        numpy array of shape [num_windows, window_size]
    """
    # Calculate number of windows
    n = len(returns)
    num_windows = ((n - window_size) // step_size) + 1
    
    # Create windows
    windows = np.zeros((num_windows, window_size))
    for i in range(num_windows):
        start_idx = i * step_size
        end_idx = start_idx + window_size
        windows[i] = returns[start_idx:end_idx]
    
    return windows


def get_raw_window(returns, window_idx, history_len, predict_len):
    """Get the raw returns window that corresponds exactly to a wavelet window index.
    
    Args:
        returns: The full sequence of raw returns
        window_idx: The index of the window (same as used for wavelet windows)
        history_len: Length of history sequence (e.g. 50)
        predict_len: Length of prediction sequence (e.g. 20)
    
    Returns:
        history_returns: numpy array of shape [history_len]
        future_returns: numpy array of shape [predict_len]
    """
    # The window_idx corresponds to the START of the window in the original sequence
    # Total window length is history_len + predict_len (e.g. 70)
    start_idx = window_idx  # This is the same index used when creating wavelet windows
    end_idx = start_idx + history_len + predict_len
    
    window_returns = returns[start_idx:end_idx]
    history_returns = window_returns[:history_len]
    future_returns = window_returns[history_len:history_len+predict_len]
    
    return history_returns, future_returns


def predict_and_save_inline(
    checkpoint_path: str,
    history_len: int = 50,
    predict_len: int = 20,
    input_dim: int = 1,
    window_configs: list = None,
    num_diffusion_timesteps: int = 500
):
    """Generate predictions for multiple windows and plot comparisons."""
    ensure_output_dirs()
    device = torch.device(CONFIG.get("device", "cuda" if torch.cuda.is_available() else "cpu"))
    print(f"🖥️ Using device: {device}")

    # If no explicit window_configs passed, use defaults from CONFIG
    if window_configs is None:
        regular = CONFIG.get("regular_samples", 100)
        high = CONFIG.get("high_samples", 300)
        nreg = CONFIG.get("num_regular_windows", 1)

        window_configs = [(regular, "regular_sample") for _ in range(nreg)]
        if CONFIG.get("include_high_sample", True):
            window_configs.append((high, "high_sample"))

    num_windows = len(window_configs)
    guidance_w = CONFIG.get("classifier_free_guidance_weight", 3.0)
    wavelet = CONFIG.get("wavelet", "db4")
    level = CONFIG.get("wavelet_level", 4)  # 4 levels → 5 bands

    print(f"📊 Generating predictions for {num_windows} windows:")
    for i, (paths, desc) in enumerate(window_configs):
        print(f"  Window {i}: {paths} paths ({desc})")

    print("📂 Loading checkpoint:", checkpoint_path)
    ckpt = torch.load(checkpoint_path, map_location=device)
    print("✅ Checkpoint loaded successfully")
    print("📦 Checkpoint keys:", list(ckpt.keys()))
    print("\n📊 Checkpoint configuration:")
    for key, value in ckpt["cfg"].items():
        print(f"  - {key}: {value}")

    # ─────────────────────────────────────────────────────────────────────────
    # 1) Load train‐wavelet means & stds (used to unnormalize later)
    # ─────────────────────────────────────────────────────────────────────────
    norm_folder = CONFIG["wavelets_path"]
    wavelet_means, wavelet_stds = load_norm_factors(norm_folder, level)
    print(f"Loaded wavelet means/stds shapes: {wavelet_means.shape}, {wavelet_stds.shape}")

    # ─────────────────────────────────────────────────────────────────────────
    # 2) Load TEST‐wavelet windows (normalized) directly from .pt
    # ─────────────────────────────────────────────────────────────────────────
    test_tensor = load_folder_as_tensor(
        root_folder = None,
        precompute_wavelets_path = CONFIG["test_data_path"],  # data/wavelets/test wavelet
        wavelet = wavelet,
        level = level
    )
    print(f"📊 Test tensor shape (precomputed test windows): {tuple(test_tensor.shape)}")
    print(f"📊 Test tensor range: [{test_tensor.min():.3f}, {test_tensor.max():.3f}]")

    # ─────────────────────────────────────────────────────────────────────────
    # 3) Load and prepare raw test returns
    # ─────────────────────────────────────────────────────────────────────────
    test_df = pd.read_csv(CONFIG["test_returns_path"])
    test_returns = test_df['Close'].values
    
    # Print information about available windows
    total_possible_windows = len(test_returns) - (history_len + predict_len) + 1
    print("\n📊 Data Information:")
    print(f"  Total returns in sequence: {len(test_returns)}")
    print(f"  Maximum possible windows: {total_possible_windows}")
    print(f"  Actual windows in wavelet tensor: {len(test_tensor)}")
    
    if total_possible_windows < len(test_tensor):
        raise ValueError(f"Not enough returns data ({len(test_returns)}) to create {len(test_tensor)} windows of size {history_len + predict_len}")

    # Create windows from raw returns
    total_window_size = history_len + predict_len
    test_windows = create_windows(test_returns, total_window_size)
    print(f"📊 Created {len(test_windows)} windows from raw returns")
    
    if len(test_windows) != len(test_tensor):
        print(f"⚠️ Warning: Number of windows mismatch! Raw returns: {len(test_windows)}, Wavelet tensor: {len(test_tensor)}")
        print("This might indicate the wavelet windows were created differently.")

    # ─────────────────────────────────────────────────────────────────────────
    # 4) Build dataset that splits each test window → (history, future)
    # ─────────────────────────────────────────────────────────────────────────
    test_dataset = WaveletSlidingWindowDataset(test_tensor, history_len, predict_len)
    total_windows = len(test_dataset)
    print(f"📊 Total available windows in *test* set: {total_windows}")

    window_indices = torch.randperm(total_windows)[:num_windows]
    print(f"📊 Randomly selected window indices: {window_indices.tolist()}")

    # ─────────────────────────────────────────────────────────────────────────
    # 5) Prepare output folder for predictions
    # ─────────────────────────────────────────────────────────────────────────
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_root = os.path.join(CONFIG["prediction_output_dir"], f"pred_{timestamp}")
    os.makedirs(save_root, exist_ok=True)
    print(f"\n📁 Saving predictions under: {save_root}")

    # ─────────────────────────────────────────────────────────────────────────
    # 6) Instantiate the ScoreTransformer model & SDE
    # ─────────────────────────────────────────────────────────────────────────
    model_net = ScoreTransformerNet(
        input_dim=input_dim,
        history_len=history_len,
        predict_len=predict_len,
        model_dim=CONFIG["model_dim"],
        num_heads=CONFIG["num_heads"],
        num_layers=CONFIG["num_layers"],
        wavelet_levels=level,
        mlp_ratio=CONFIG["mlp_ratio"],
        drop_rate=CONFIG["drop_rate"],
        attn_drop_rate=CONFIG["attn_drop_rate"]
    ).to(device)

    print("\n📊 Model Architecture:")
    print(f"  - Input dimension: {input_dim}")
    print(f"  - Wavelet bands (L+1): {level+1} [cD1,cD2,cD3,cD4,cA4]")
    print(f"  - Wavelet levels: {level}")
    print(f"  - Model dimension: {model_net.model_dim}")
    print(f"  - Number of heads: {CONFIG['num_heads']}")
    print(f"  - Number of layers: {CONFIG['num_layers']}")

    # ─────────────────────────────────────────────────────────────────────────
    # 7) Load weights from checkpoint (filter unmatched keys)
    # ─────────────────────────────────────────────────────────────────────────
    checkpoint_dict = ckpt["model"] if "model" in ckpt else ckpt
    model_dict = model_net.state_dict()
    filtered = {k: v for (k, v) in checkpoint_dict.items() if k in model_dict}
    model_dict.update(filtered)
    model_net.load_state_dict(model_dict)
    print("✅ Model weights loaded (filtered).")
    model_net.eval()

    ###############################
    # 8) Initialize the SDE model #
    ###############################
    sde_model = VPSDE()
    print("✅ SDE initialized")

    # 8) Loop over each selected test window & sample multiple paths
    all_stats = []
    for window_idx, (paths_per_window, desc) in enumerate(window_configs):
        idx = window_indices[window_idx].item()
        print(f"\n🔄 Window index {idx} ({desc})")
        window_dir = os.path.join(save_root, f"window_{idx}_{desc}")
        os.makedirs(window_dir, exist_ok=True)

        # Get raw returns window using EXACT same index as wavelet window
        history_returns, future_returns = get_raw_window(
            test_returns, idx, history_len, predict_len
        )
        
        # Print window information for verification
        print("\nWindow Information:")
        print(f"  Start index in sequence: {idx}")
        print(f"  End index in sequence: {idx + history_len + predict_len}")
        print(f"  History returns range: [{history_returns.min():.4f}, {history_returns.max():.4f}]")
        print(f"  Future returns range: [{future_returns.min():.4f}, {future_returns.max():.4f}]")

        # 8a) Load normalized‐wavelet (history, future) for this test window
        hist_norm, fut_norm = test_dataset[idx]
        hist_norm = hist_norm.to(device)   # [history_len, 5, feat_dim]
        fut_norm  = fut_norm.to(device)    # [predict_len,  5, feat_dim]

        # 8b) Expand so we can sample B paths in parallel
        hist_batch = hist_norm.unsqueeze(0).expand(paths_per_window, -1, -1, -1)  # [B, H, 5, D]

        print(f"  History_norm expanded shape: {tuple(hist_batch.shape)}")
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # 8c) Run reverse‐SDE sampler to get B normalized‐wavelet futures
        x_norm = reverse_sde_sampler(
            model_net=model_net,
            sde_model=sde_model,
            history=hist_batch,
            num_steps=CONFIG["num_diffusion_timesteps"],
            guidance_weight=guidance_w,
            device=device
        )  # [B, predict_len, 5, feat_dim]
        print(f"  Sampled normalized shape: {tuple(x_norm.shape)}")

        x_norm_np = x_norm.cpu().numpy()  # [B, T, 5, feat_dim]
        fut_norm_np = fut_norm.cpu().numpy()  # [T, 5, feat_dim]

        # 8d) Unnormalize wavelet coefficients
        x_unnorm = unnormalize_wavelet(x_norm_np, wavelet_means, wavelet_stds)  # [B, T, 5, feat_dim]
        fut_unnorm = unnormalize_wavelet(fut_norm_np[None], wavelet_means, wavelet_stds)  # [1, T, 5, feat_dim]
        print(f"  Unnormalized wavelet shape: {x_unnorm.shape}")

        # Plot wavelet comparisons
        plot_wavelet_comparison(x_norm_np, x_unnorm, fut_norm_np[None], fut_unnorm, window_dir)

        # Convert wavelets to returns
        B, T, Lp, Dfeat = x_unnorm.shape
        flat_wave = x_unnorm.reshape(B * T, Lp * Dfeat)                         # [B*T, 5*feat_dim]
        recon_flat = test_dataset.inverse_transform(flat_wave)                  # [B*T, feat_dim]
        recon = recon_flat.reshape(B, T, -1)                                     # [B, T, feat_dim]
        recon_np = recon.cpu().numpy().squeeze()                                  # [B, T]

        # Plot price comparisons and save statistics
        window_stats = plot_price_comparison(
            history_returns=history_returns,
            future_returns=future_returns,
            pred_returns=recon_np,
            window_dir=window_dir
        )
        
        if window_stats is not None:
            window_stats['window_idx'] = idx
            window_stats['window_type'] = desc
            all_stats.append(window_stats)

        # Save raw arrays
        np.save(os.path.join(window_dir, "x_norm.npy"), x_norm_np)
        np.save(os.path.join(window_dir, "x_unnorm.npy"), x_unnorm)
        np.save(os.path.join(window_dir, "reconstructed_returns.npy"), recon_np)
        print(f"  Saved arrays to {window_dir}")

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Save aggregate statistics
    if all_stats:
        stats_df = pd.DataFrame(all_stats)
        stats_df.to_csv(os.path.join(save_root, "prediction_statistics.csv"), index=False)
        print("\n✅ Saved aggregate statistics to prediction_statistics.csv")

    print("\n✅ Prediction generation and plotting complete!")


if __name__ == "__main__":
    predict_and_save_inline(
        checkpoint_path=find_latest_checkpoint(),
        history_len=CONFIG["history_len"],
        predict_len=CONFIG["predict_len"],
        input_dim=CONFIG.get("input_dim", 1),
        window_configs=None,
        num_diffusion_timesteps=CONFIG["num_diffusion_timesteps"]
    )
