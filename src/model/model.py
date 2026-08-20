# src/model/model.py

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)                     # (max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)
        div = torch.exp(torch.arange(0, d_model, 2).float()     # (d_model/2,)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)                      # apply sin to even indices
        pe[:, 1::2] = torch.cos(pos * div)                      # apply cos to odd indices
        self.register_buffer('pe', pe.unsqueeze(0))             # (1, max_len, d_model)

    def forward(self, length):
        # Returns the first `length` positions: (1, length, d_model)
        return self.pe[:, :length]


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(1, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t):
        # t: (B,) or (B, 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)        # (B, 1)
        out = self.proj(t)             # (B, dim)
        return out.unsqueeze(1)        # → (B, 1, dim)


class ScoreTransformerNet(nn.Module):
    def __init__(
        self,
        input_dim,        # Original feature dimension (feat_dim)
        history_len,      # Length of history sequence (T_history)
        predict_len,      # Length of future/prediction sequence (T_future)
        model_dim=256,
        num_heads=8,
        num_layers=4,
        wavelet_levels=3,
        mlp_ratio=4.0,
        drop_rate=0.1,
        attn_drop_rate=0.1
    ):
        super().__init__()

        # Basic dimensions
        self.history_len = history_len
        self.predict_len = predict_len
        self.L = wavelet_levels            # number of wavelet levels minus 1
        self.feat_dim = input_dim          # original feature dimension per wavelet band
        self.model_dim = model_dim

        # ---------------------------------------------------------
        # 1) Shared per‐scale projections (instead of separate lists)
        # ---------------------------------------------------------
        self.proj_hist = nn.Linear(self.feat_dim, model_dim)
        self.proj_fut = nn.Linear(self.feat_dim, model_dim)

        # ---------------------------------------------------------
        # 2) Positional embeddings for history and future
        # ---------------------------------------------------------
        self.pos_hist = PositionalEncoding(model_dim, max_len=history_len)
        self.pos_fut = PositionalEncoding(model_dim, max_len=predict_len)

        # ---------------------------------------------------------
        # 3) Scale embeddings (one embedding per wavelet band)
        # ---------------------------------------------------------
        self.scale_emb = nn.Embedding(self.L + 1, model_dim)

        # ---------------------------------------------------------
        # 4) Time embedding to condition on t
        # ---------------------------------------------------------
        self.time_emb = TimeEmbedding(model_dim)

        # ---------------------------------------------------------
        # 5) Shared Transformer encoder for all wavelet scales
        # ---------------------------------------------------------
        from torch.nn import TransformerEncoder, TransformerEncoderLayer

        enc_layer = TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dim_feedforward=int(model_dim * mlp_ratio),
            dropout=drop_rate,
            activation='gelu',
            batch_first=True
        )
        self.shared_encoder = TransformerEncoder(enc_layer, num_layers)

        # ---------------------------------------------------------
        # 6) Fuse per‐scale outputs back into a single (B, T, model_dim)
        # ---------------------------------------------------------
        self.fuse_hist = nn.Linear((self.L + 1) * model_dim, model_dim)
        self.fuse_fut = nn.Linear((self.L + 1) * model_dim, model_dim)

        # ---------------------------------------------------------
        # 7) Decoder: stack of TransformerDecoderLayer (self‐attn → cross‐attn → FFN)
        # ---------------------------------------------------------
        from torch.nn import TransformerDecoderLayer

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                d_model=model_dim,
                nhead=num_heads,
                dim_feedforward=int(model_dim * mlp_ratio),
                dropout=drop_rate,
                activation='gelu',
                batch_first=True
            )
            for _ in range(num_layers)
        ])

        # ---------------------------------------------------------
        # 8) CFG tokens: one mask_token for random masking, one null_cond for classifier‐free drop
        #    - mask_token: shape (1, 1, model_dim) to broadcast over masked positions
        #    - null_cond: shape (1, history_len, model_dim) to replace an entire sequence
        # ---------------------------------------------------------
        self.mask_token = nn.Parameter(torch.randn(1, 1, model_dim))
        self.null_cond = nn.Parameter(torch.randn(1, history_len, model_dim))

        # ---------------------------------------------------------
        # 9) Final projection: model_dim → (L+1)*feat_dim, then reshape to (B, T, L+1, feat_dim)
        # ---------------------------------------------------------
        self.output_proj = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, (self.L + 1) * self.feat_dim)
        )

    def _encode(self, bands, is_hist: bool):
        """
        Internal encoding for either historical or future wavelet bands.

        Args:
            bands: either
                - (B, T, feat_dim) if no explicit wavelet decomposition passed, OR
                - (B, T, L+1, feat_dim) if already decomposed into (L+1) bands.
            is_hist: True → use history projection + fuse_hist; False → use future projection + fuse_fut

        Returns:
            Tensor of shape (B, T, model_dim), the fused encoding across all (L+1) wavelet scales.
        """
        if bands.dim() == 3:
            # Expand to 4D: (B, T, L+1, feat_dim)
            bands = bands.unsqueeze(2).expand(-1, -1, self.L + 1, -1)

        B, T, _, _ = bands.shape
        device = bands.device

        # Select positional embedding list & fusion layer
        if is_hist:
            pos_embed = self.pos_hist(T).to(device)    # (1, T, model_dim)
        else:
            pos_embed = self.pos_fut(T).to(device)     # (1, T, model_dim)
        pos = pos_embed.expand(B, -1, -1)             # (B, T, model_dim)

        outs = []
        for j in range(self.L + 1):
            # Extract the j-th wavelet band: (B, T, feat_dim)
            x_j = bands[:, :, j, :]

            # Apply shared projection
            if is_hist:
                x_j = self.proj_hist(x_j)              # → (B, T, model_dim)
            else:
                x_j = self.proj_fut(x_j)               # → (B, T, model_dim)

            # Create scale embedding for band j: (T, model_dim) → expand to (B, T, model_dim)
            scale = self.scale_emb(
                torch.full((T,), j, device=device, dtype=torch.long)
            )                                          # (T, model_dim)
            scale = scale.unsqueeze(0).expand(B, -1, -1)  # (B, T, model_dim)

            # Add positional + scale embeddings
            x_j = x_j + pos + scale                     # (B, T, model_dim)

            # Encode via shared TransformerEncoder
            enc_j = self.shared_encoder(x_j)            # (B, T, model_dim)
            outs.append(enc_j)

        # Concatenate along model_dim: (B, T, (L+1)*model_dim)
        cat = torch.cat(outs, dim=-1)
        # Fuse back down to (B, T, model_dim)
        return self.fuse_hist(cat) if is_hist else self.fuse_fut(cat)

    def forward(
        self,
        x_t,                 # future input: (B, T_future, feat_dim) or (B, T_future, L+1, feat_dim)
        x_hist,              # history input: (B, T_history, feat_dim) or (B, T_history, L+1, feat_dim)
        t,                   # timestep conditioning: (B,) or (B,1)
        cond_drop_prob=0.1,  # classifier‐free drop probability (per‐sample)
        mask_ratio=0.15      # random masking ratio within each sequence
    ):
        B = x_t.size(0)
        T_fut = x_t.size(1)
        T_hist = x_hist.size(1)

        # Ensure both x_t and x_hist are 4D: (B, T, L+1, feat_dim)
        if x_t.dim() == 3:
            x_t = x_t.unsqueeze(2).expand(-1, -1, self.L + 1, -1)
        if x_hist.dim() == 3:
            x_hist = x_hist.unsqueeze(2).expand(-1, -1, self.L + 1, -1)

        # 1) Encode history and future separately
        enc_hist = self._encode(x_hist, is_hist=True)   # (B, T_history, model_dim)
        enc_fut = self._encode(x_t, is_hist=False)      # (B, T_future, model_dim)

        # 2) Random masking within each sequence (mask_ratio% of time steps per sample)
        if self.training and mask_ratio > 0.0:
            num_h = int(T_hist * mask_ratio)
            num_f = int(T_fut * mask_ratio)

            for i in range(B):
                # Shuffle time indices and pick the first num_h to mask for history
                hi = torch.randperm(T_hist, device=enc_hist.device)[:num_h]
                # Shuffle time indices and pick the first num_f to mask for future
                fi = torch.randperm(T_fut, device=enc_fut.device)[:num_f]

                # Broadcast mask_token over those time positions
                enc_hist[i, hi] = self.mask_token.to(dtype=enc_hist.dtype)
                enc_fut[i, fi] = self.mask_token.to(dtype=enc_fut.dtype)

        # 3) Add time embedding (shared between history & future)
        t_emb = self.time_emb(t.to(enc_hist.device))      # (B, 1, model_dim)
        enc_hist = enc_hist + t_emb.expand(-1, T_hist, -1)  # (B, T_history, model_dim)
        enc_fut = enc_fut + t_emb.expand(-1, T_fut, -1)     # (B, T_future, model_dim)

        # 4) Classifier‐free guidance: per‐sample random drop of the entire history encoder
        if self.training and cond_drop_prob > 0.0:
            # Draw a coin flip for each sample: True means "drop condition" for that sample
            mask_cf = torch.rand(B, device=enc_hist.device) < cond_drop_prob
            for i in range(B):
                if mask_cf[i]:
                    # Replace enc_hist[i] (shape (T_history, model_dim)) with null_cond (squeezed to (T_history, model_dim))
                    enc_hist[i] = self.null_cond.squeeze(0)

        # 5) Decoder stack: masked self‐attention → cross‐attention → FFN (handled internally by TransformerDecoderLayer)
        x = enc_fut  # initial decoder input (B, T_future, model_dim)
        for dec_layer in self.decoder_layers:
            # Note: since batch_first=True, arguments are (tgt, memory)
            x = dec_layer(tgt=x, memory=enc_hist)  # (B, T_future, model_dim)

        # 6) Project back to wavelet‐band scores: (B, T_future, (L+1)*feat_dim) → reshape → (B, T_future, L+1, feat_dim)
        proj = self.output_proj(x)                           # (B, T_future, (L+1)*feat_dim)
        out = proj.view(B, T_fut, self.L + 1, self.feat_dim)  # → (B, T_future, L+1, feat_dim)

        return out
