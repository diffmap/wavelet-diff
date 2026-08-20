import numpy as np
import pytest

from src.model.baselines import last_return, zero_return
from src.model.evaluation import rolling_origin_evaluate


def test_rolling_evaluation_passes_only_past_values() -> None:
    series = np.arange(12, dtype=float)
    seen_histories = []

    def forecast(history: np.ndarray) -> np.ndarray:
        seen_histories.append(history.copy())
        return zero_return(history, horizon=2)

    records, aggregate = rolling_origin_evaluate(
        series,
        forecast,
        history_len=4,
        horizon=2,
        start=4,
        stop=6,
    )

    assert [record.origin for record in records] == [4, 5, 6]
    np.testing.assert_array_equal(seen_histories[0], [0, 1, 2, 3])
    np.testing.assert_array_equal(seen_histories[-1], [2, 3, 4, 5])
    assert aggregate["n_origins"] == 3.0


def test_rolling_evaluation_accepts_deterministic_baseline() -> None:
    series = np.arange(10, dtype=float)

    _, aggregate = rolling_origin_evaluate(
        series,
        lambda history: last_return(history, horizon=2),
        history_len=3,
        horizon=2,
        start=3,
        stop=3,
    )

    assert aggregate["mae"] == 1.5
    assert aggregate["rmse"] == np.sqrt(2.5)


def test_rolling_evaluation_rejects_wrong_forecast_length() -> None:
    with pytest.raises(ValueError, match="must return"):
        rolling_origin_evaluate(
            np.arange(8, dtype=float),
            lambda history: np.zeros((2, 3)),
            history_len=3,
            horizon=2,
            start=3,
            stop=3,
        )
