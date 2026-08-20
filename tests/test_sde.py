import torch

from src.model.sde import VPSDE


def test_forward_process_near_zero_time_preserves_signal() -> None:
    sde = VPSDE(bmin=0.1, bmax=20.0)
    x = torch.randn(4, 3, 2, 1)
    t = torch.full((4,), 1e-5)

    mu, std = sde.p(x, t)

    torch.testing.assert_close(mu, x, atol=1e-3, rtol=1e-3)
    assert std.max() < 0.05


def test_forward_process_near_full_time_destroys_signal() -> None:
    sde = VPSDE(bmin=0.1, bmax=20.0)
    x = torch.randn(4, 3, 2, 1)
    t = torch.full((4,), 1.0)

    mu, std = sde.p(x, t)

    assert mu.abs().max() < 0.05
    assert std.min() > 0.95


def test_alpha_is_monotonically_decreasing_in_t() -> None:
    sde = VPSDE(bmin=0.1, bmax=20.0)
    t = torch.linspace(0.0, 1.0, steps=10)

    alpha = sde.alpha(t)

    assert torch.all(alpha[1:] <= alpha[:-1])
