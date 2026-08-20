# src/model/data.py

import os
import torch
import numpy as np
from torch.utils.data import Dataset


def load_folder_as_tensor(root_folder, precompute_wavelets_path, wavelet, level):
    """
    Load a precomputed wavelet‐windows .pt file from `precompute_wavelets_path`.
    If `root_folder` is not None, read CSV(s) to compute wavelets → save .pt → return tensor.

    Returns:
      A torch.Tensor of shape [N_windows, total_len, (level+1), feat_dim],
      where total_len = history_len + predict_len (set elsewhere).
    """
    L = level
    if precompute_wavelets_path is not None:
        # First try to load the 'train' version:
        fname = os.path.join(precompute_wavelets_path, f"level{L}_swt_train_windows_norm.pt")
        if not os.path.isfile(fname):
            # If not found, try the 'test' version:
            fname = os.path.join(precompute_wavelets_path, f"level{L}_swt_test_windows_norm.pt")
            if not os.path.isfile(fname):
                raise FileNotFoundError(
                    f"[data.py] Cannot find\n"
                    f"  {os.path.join(precompute_wavelets_path, f'level{L}_swt_train_windows_norm.pt')}\n"
                    f"or\n"
                    f"  {os.path.join(precompute_wavelets_path, f'level{L}_swt_test_windows_norm.pt')}"
                )
        tensor = torch.load(fname)  # could be [N, total_len, (L+1)*feat_dim] or [N, total_len, L+1, feat_dim]
        if tensor.dim() == 3:
            # If shape is [N, T, D], where D = (L+1) * feat_dim, reshape into 4D
            N, T, D = tensor.shape
            if D % (L + 1) != 0:
                raise ValueError(f"[data.py] Cannot reshape last dim: level+1={L+1}, D={D}")
            feat_dim = D // (L + 1)
            tensor = tensor.view(N, T, L + 1, feat_dim)
        return tensor

    else:
        raise NotImplementedError("[data.py] CSV-to-wavelet logic is not used by sampler.")


class WaveletSlidingWindowDataset(Dataset):
    """
    Given a tensor of shape [N_windows, total_len, (L+1), feat_dim],
    this Dataset returns (history, future) pairs:
      - history: [history_len, L+1, feat_dim]
      - future : [predict_len,  L+1, feat_dim]
    """

    def __init__(self, tensor, history_len, predict_len):
        super().__init__()
        self.tensor = tensor  # [N_windows, total_len, L+1, feat_dim]
        N, T, Lp, D = tensor.shape
        expected_T = history_len + predict_len
        if T != expected_T:
            raise ValueError(f"[data.py] Dataset init: expected total_len={expected_T}, but got T={T}")
        self.history_len = history_len
        self.predict_len = predict_len
        self.Lp = Lp
        self.D = D  # original per-band feature dimension (should be 1 for your returns)

    def __len__(self):
        return self.tensor.size(0)

    def __getitem__(self, idx):
        """
        Returns:
          hist: [history_len, L+1, feat_dim]
          fut : [predict_len,  L+1, feat_dim]
        """
        window = self.tensor[idx]  # [T, L+1, D]
        hist = window[: self.history_len]
        fut = window[self.history_len : self.history_len + self.predict_len]
        return hist, fut

    def inverse_transform(self, flat_wave):
        """
        Simplified inverse_transform: use only the final (approximation) band 
        as the "reconstructed" return. This guarantees a shape of [M, feat_dim=1].

        Accepts either:
          - flat_wave: numpy array of shape [M, (L+1)*feat_dim]
          - flat_wave: torch.Tensor of shape [M, (L+1)*feat_dim]

        We simply take the last column (corresponding to the highest‐level approximation band).
        Returns: a torch.Tensor of shape [M, feat_dim].
        """
        # If incoming is a torch.Tensor, convert to numpy first:
        if isinstance(flat_wave, torch.Tensor):
            flat_wave = flat_wave.detach().cpu().numpy()

        # Now flat_wave is a NumPy array, shape [M, (L+1)*feat_dim].
        # We pick the last column (approximation band) as the “return.”
        approx_band = flat_wave[:, -1]  # shape [M]

        # Reshape to [M, 1] and turn into float32 torch.Tensor:
        return torch.from_numpy(approx_band.reshape(-1, 1).astype(np.float32))
