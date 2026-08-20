from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ─────────────────────────────────────────────────────────────────────────────
# 1) Load the “true” normalized test‐wavelet tensor (70 time‐steps per window)
# ─────────────────────────────────────────────────────────────────────────────
PT_PATH = PROJECT_ROOT / "data" / "wavelets" / "test wavelet" / "level4_swt_test_windows_norm.pt"
tensor = torch.load(PT_PATH)   # shape [N_full, 70, 5, 1] or [N_full, 70, 5]
if tensor.ndim == 3:
    N_full, T_full, D = tensor.shape
    tensor = tensor.view(N_full, T_full, 5, 1)

true_norm = tensor.squeeze(-1).cpu().numpy()  # → [N_full, 70, 5]
assert true_norm.shape[1] == 70, f"Expected 70 steps, got {true_norm.shape[1]}"

# ─────────────────────────────────────────────────────────────────────────────
# 2) Load the model’s predicted normalized wavelets (only 20‐step future)
# ─────────────────────────────────────────────────────────────────────────────
PRED_PATH = PROJECT_ROOT / "outputs" / "predictions" / "pred_20250607_183238" / "window_248_regular_sample" / "x_norm.npy"
x_norm = np.load(PRED_PATH)   # shape [N_pred, 20, 5, 1] or [N_pred, 20, 5]
if x_norm.ndim == 4:
    N_pred, T_pred, num_bands, _ = x_norm.shape
    x_norm = x_norm.squeeze(-1)   # → [N_pred, 20, 5]
else:
    N_pred, T_pred, num_bands = x_norm.shape

pred_norm = x_norm   # → [N_pred, 20, 5]
assert T_pred == 20, f"Expected 20 time‐steps, got {T_pred}"
assert num_bands == 5, f"Expected 5 bands, got {num_bands}"

# ─────────────────────────────────────────────────────────────────────────────
# 3) Extract window index 986 and split into true_fut (last 20) and pred_fut
# ─────────────────────────────────────────────────────────────────────────────
idx = 986
assert idx < true_norm.shape[0], f"True index {idx} out of range (max {true_norm.shape[0]-1})"
assert idx < pred_norm.shape[0], f"Pred index {idx} out of range (max {pred_norm.shape[0]-1})"

# True window: 70×5
true_window = true_norm[idx]   # → [70, 5]
# Predicted future: 20×5
pred_fut = pred_norm[idx]      # → [20, 5]

# Last 20 steps of the true window:
true_fut = true_window[50:, :]  # → [20, 5]
assert true_fut.shape == (20, 5)

# ─────────────────────────────────────────────────────────────────────────────
# 4) Plot the 20‐step “future” for each band side by side
# ─────────────────────────────────────────────────────────────────────────────
band_names = ["cD1", "cD2", "cD3", "cD4", "cA4"]
t_fut = np.arange(20)

# Compute per‐band MAE over those 20 steps
mae_per_band = np.mean(np.abs(true_fut - pred_fut), axis=0)  # → [5]

# Determine a common y‐range
global_min = min(true_fut.min(), pred_fut.min())
global_max = max(true_fut.max(), pred_fut.max())

for b, band in enumerate(band_names):
    plt.figure(figsize=(6, 3.5))

    # True future (solid blue)
    plt.plot(
        t_fut,
        true_fut[:, b],
        color="tab:blue",
        linewidth=1.5,
        label=f"True (window {idx}, last 20)"
    )
    # Predicted future (dashed orange)
    plt.plot(
        t_fut,
        pred_fut[:, b],
        color="tab:orange",
        linewidth=1.5,
        linestyle="--",
        label=f"Pred (window {idx}, predicted 20)"
    )

    # Annotate MAE
    plt.text(
        0.02 * 20,
        global_max * 0.9,
        f"MAE: {mae_per_band[b]:.4f}",
        fontsize=9,
        bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray")
    )

    plt.title(f"Window {idx} • Band {band} Over Future 20 Steps")
    plt.xlabel("Time step (0 … 19)")
    plt.ylabel("Z‐score Value")
    plt.ylim(global_min - 0.05, global_max + 0.05)
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()
