import numpy as np

from src.model.predictandsave import compute_crps


def test_crps_is_zero_for_a_perfect_deterministic_ensemble() -> None:
    truth = np.array([1.0, -2.0, 0.5])
    samples = np.tile(truth, (20, 1))

    assert compute_crps(samples, truth) == 0.0


def test_crps_is_nonnegative_for_a_noisy_ensemble() -> None:
    rng = np.random.default_rng(0)
    truth = np.zeros(5)
    samples = rng.normal(loc=0.0, scale=1.0, size=(200, 5))

    assert compute_crps(samples, truth) >= 0.0


def test_crps_increases_with_bias_at_fixed_spread() -> None:
    rng = np.random.default_rng(0)
    truth = np.zeros(5)
    unbiased = rng.normal(loc=0.0, scale=1.0, size=(200, 5))
    biased = unbiased + 3.0

    assert compute_crps(biased, truth) > compute_crps(unbiased, truth)
