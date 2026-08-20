import torch

from src.model.model import ScoreTransformerNet


def _build_tiny_model() -> ScoreTransformerNet:
    return ScoreTransformerNet(
        input_dim=1,
        history_len=4,
        predict_len=2,
        model_dim=8,
        num_heads=2,
        num_layers=1,
        wavelet_levels=1,
        mlp_ratio=2.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
    )


def test_forward_output_shape_matches_future_window() -> None:
    model = _build_tiny_model()
    batch, history_len, predict_len, bands, feat_dim = 3, 4, 2, 2, 1

    x_hist = torch.randn(batch, history_len, bands, feat_dim)
    x_t = torch.randn(batch, predict_len, bands, feat_dim)
    t = torch.rand(batch, 1)

    out = model(x_t, x_hist, t, cond_drop_prob=0.0, mask_ratio=0.0)

    assert out.shape == (batch, predict_len, bands, feat_dim)


def test_forward_accepts_3d_bands_by_broadcasting() -> None:
    model = _build_tiny_model()
    batch, history_len, predict_len, feat_dim = 2, 4, 2, 1

    x_hist = torch.randn(batch, history_len, feat_dim)
    x_t = torch.randn(batch, predict_len, feat_dim)
    t = torch.rand(batch, 1)

    out = model(x_t, x_hist, t, cond_drop_prob=0.0, mask_ratio=0.0)

    assert out.shape == (batch, predict_len, 2, feat_dim)
