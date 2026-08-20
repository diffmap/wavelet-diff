# src/model/trainer.py

import os
import copy
import numpy as np
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader, Sampler
from torch.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from src.model import config
from src.model import model as model_module
from src.model import sde
from src.model import data


class EpochRandomSubsetSampler(Sampler):
    """
    Every epoch, returns a fresh random subset of `samples_per_epoch` indices 
    from [0 … dataset_size). DataLoader will call __iter__() at the start 
    of each epoch, so you see a new subset each time.
    """
    def __init__(self, dataset_size: int, samples_per_epoch: int):
        self.dataset_size = dataset_size
        self.samples_per_epoch = samples_per_epoch

    def __len__(self):
        # Number of samples per “epoch”
        return self.samples_per_epoch

    def __iter__(self):
        # Return a brand‐new random list of indices each epoch
        idxs = torch.randperm(self.dataset_size)[: self.samples_per_epoch].tolist()
        return iter(idxs)


def update_ema(ema_model, model, decay):
    """EMA update: ema_param = decay * ema_param + (1 - decay) * model_param"""
    with torch.no_grad():
        m = dict(model.named_parameters())
        for n, p in ema_model.named_parameters():
            p.data.mul_(decay).add_(m[n].data, alpha=1.0 - decay)


def get_dynamic_mask_ratio(ep, ne, mn, mx):
    """Cosine-anneal mask ratio between mn and mx over ne epochs, at epoch ep."""
    return mn + (mx - mn) * 0.5 * (1 + np.cos(np.pi * ep / ne))


def train():
    config.ensure_output_dirs()

    device      = config.CONFIG.get("device", "cuda")
    history_len = config.CONFIG["history_len"]
    predict_len = config.CONFIG["predict_len"]
    L           = config.CONFIG["wavelet_level"]   # This is the wavelet level
    λ_wave      = config.CONFIG.get("lambda_wave", 0.01)
    use_amp     = config.CONFIG.get("use_amp", True)

    # 1) Load precomputed 4-D wavelet windows
    train_tensor = data.load_folder_as_tensor(
        config.CONFIG["train_data_path"],
        precompute_wavelets_path=config.CONFIG["wavelets_path"],
        wavelet=config.CONFIG["wavelet"],
        level=L
    )
    N_total = train_tensor.shape[0]
    print(f"[trainer.py] Loaded train_tensor shape: {tuple(train_tensor.shape)} (N={N_total})")

    dataset = data.WaveletSlidingWindowDataset(
        train_tensor,
        history_len,
        predict_len
    )
    print(f"[trainer.py] Dataset size (windows): {len(dataset)}")

    # Use a sampler to draw only samples_per_epoch each epoch
    samples_per_epoch = config.CONFIG["samples_per_epoch"]  # e.g. 300
    sampler = EpochRandomSubsetSampler(N_total, samples_per_epoch)

    loader = DataLoader(
        dataset,
        batch_size=config.CONFIG["batch_size"],
        sampler=sampler,                          # ← use custom sampler
        pin_memory=config.CONFIG["pin_memory"],
        num_workers=config.CONFIG["num_workers"],
        prefetch_factor=(
            config.CONFIG["prefetch_factor"] if config.CONFIG["num_workers"] > 0 else None
        ),
        persistent_workers=True                    # keep workers alive across epochs
    )

    # Print how many batches per epoch
    print(f"[trainer.py] Batches per epoch (with {samples_per_epoch} samples): {len(loader)}")

    grad_accum_steps = config.CONFIG["grad_accum_steps"]
    effective_batch  = config.CONFIG["batch_size"] * grad_accum_steps

    print("\nTraining Configuration:")
    print(f"  - Total windows:       {N_total}")
    print(f"  - Samples per epoch:   {samples_per_epoch}")
    print(f"  - Batch size:          {config.CONFIG['batch_size']} (effective: {effective_batch})")
    print(f"  - Batches per epoch:   {len(loader)}")
    print(f"  - Training epochs:     {config.CONFIG['n_epochs']}\n")

    # 2) Instantiate model, EMA copy, optimizer, scheduler, scaler, and SDE
    feat_dim = train_tensor.shape[-1]
    print(f"[trainer.py] Building ScoreTransformerNet with input_dim = {feat_dim}")

    model = model_module.ScoreTransformerNet(
        input_dim      = feat_dim,
        history_len    = history_len,
        predict_len    = predict_len,
        model_dim      = config.CONFIG["model_dim"],
        num_heads      = config.CONFIG["num_heads"],
        num_layers     = config.CONFIG["num_layers"],
        wavelet_levels = L,                  # Pass L (wavelet level)
        mlp_ratio      = config.CONFIG.get("mlp_ratio", 4.0),
        drop_rate      = config.CONFIG.get("drop_rate", 0.1),
        attn_drop_rate = config.CONFIG.get("attn_drop_rate", 0.1)
    ).to(device)

    print(f"[trainer.py] Model dims → model_dim = {model.model_dim}, heads = {config.CONFIG['num_heads']}, layers = {config.CONFIG['num_layers']}")

    ema_model = copy.deepcopy(model).eval()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.CONFIG["lr"],
        weight_decay=config.CONFIG.get("weight_decay", 0.01)
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config.CONFIG["n_epochs"]
    )

    scaler = GradScaler() if (use_amp and device.startswith("cuda")) else None
    sde_model = sde.VPSDE()

    # 3) Training loop
    epoch_pbar = tqdm(range(config.CONFIG["n_epochs"]), desc="Training Progress")

    for epoch in epoch_pbar:
        model.train()
        total_loss = 0.0
        optimizer.zero_grad()

        if config.CONFIG.get("dynamic_masking", True):
            mask_ratio = get_dynamic_mask_ratio(
                epoch,
                config.CONFIG["n_epochs"],
                config.CONFIG.get("min_mask_ratio", 0.05),
                config.CONFIG.get("max_mask_ratio", 0.15)
            )
        else:
            mask_ratio = config.CONFIG["mask_ratio"]

        batch_pbar = tqdm(
            enumerate(loader),
            total=len(loader),
            desc=f"Epoch {epoch+1}",
            leave=False
        )

        for batch_idx, (hist_flat, fut_flat) in batch_pbar:
            hist = hist_flat.to(device)  # [B, history_len, L+1, feat_dim]
            fut  = fut_flat.to(device)   # [B, predict_len,   L+1, feat_dim]

            if epoch == 0 and batch_idx == 0:
                print(f"[trainer.py] First batch shapes → hist: {tuple(hist.shape)}, fut: {tuple(fut.shape)}")

            B = hist.size(0)
            t = (torch.rand(B, device=device) * 0.998 + 0.001).unsqueeze(-1)

            if use_amp and device.startswith("cuda"):
                with autocast(device_type="cuda"):
                    mu, std = sde_model.p(
                        fut,
                        t.expand(-1, config.CONFIG["predict_len"])
                    )
                    noise = torch.randn_like(std)
                    x_t   = mu + std * noise

                    score = model(
                        x_t,
                        hist,
                        t,
                        cond_drop_prob=config.CONFIG["cond_drop_prob"],
                        mask_ratio=mask_ratio
                    )
                    loss_sm = torch.mean((std * score + noise) ** 2)

                    x0      = x_t - std * score
                    ms_loss = 0.0
                    for j in range(L + 1):
                        ms_loss += (1.0 / (j + 1)) * F.mse_loss(
                            x0[:, :, j, :], fut[:, :, j, :]
                        )

                    loss = loss_sm + λ_wave * ms_loss
                    loss = loss / grad_accum_steps

                scaler.scale(loss).backward()
            else:
                mu, std = sde_model.p(
                    fut,
                    t.expand(-1, config.CONFIG["predict_len"])
                )
                noise = torch.randn_like(std)
                x_t   = mu + std * noise

                score = model(
                    x_t,
                    hist,
                    t,
                    cond_drop_prob=config.CONFIG["cond_drop_prob"],
                    mask_ratio=mask_ratio
                )
                loss_sm = torch.mean((std * score + noise) ** 2)

                x0      = x_t - std * score
                ms_loss = 0.0
                for j in range(L + 1):
                    ms_loss += (1.0 / (j + 1)) * F.mse_loss(
                        x0[:, :, j, :], fut[:, :, j, :]
                    )

                loss = loss_sm + λ_wave * ms_loss
                loss = loss / grad_accum_steps
                loss.backward()

            do_step = (
                (batch_idx + 1) % config.CONFIG["grad_accum_steps"] == 0
                or (batch_idx + 1 == len(loader))
            )
            if do_step:
                if use_amp and device.startswith("cuda"):
                    scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), config.CONFIG.get("grad_clip", 1.0))

                if use_amp and device.startswith("cuda"):
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                update_ema(ema_model, model, config.CONFIG["ema_decay"])

            total_loss += loss.item() * config.CONFIG["grad_accum_steps"]
            batch_pbar.set_postfix({
                "Loss": f"{(loss.item() * config.CONFIG['grad_accum_steps']):.4f}"
            })

        scheduler.step()
        epoch_loss = total_loss / len(loader)
        epoch_pbar.set_postfix({"Loss": f"{epoch_loss:.4f}"})

        if (epoch + 1) % config.CONFIG["checkpoint_freq"] == 0:
            ckpt_path = os.path.join(
                config.CONFIG["checkpoint_dir"],
                f"{config.CONFIG['save_name']}_ep{epoch+1}.pth"
            )
            torch.save({
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "opt": optimizer.state_dict(),
                "sched": scheduler.state_dict(),
                "cfg": config.CONFIG
            }, ckpt_path)

    print("🏁 Training complete.")


if __name__ == "__main__":
    print("🚀 Starting training")
    train()
