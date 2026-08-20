import torch
import pytest

from src.model.data import WaveletSlidingWindowDataset


def test_dataset_splits_history_and_future() -> None:
    tensor = torch.arange(2 * 7 * 5, dtype=torch.float32).reshape(2, 7, 5, 1)
    dataset = WaveletSlidingWindowDataset(tensor, history_len=5, predict_len=2)

    history, future = dataset[0]

    assert history.shape == (5, 5, 1)
    assert future.shape == (2, 5, 1)
    torch.testing.assert_close(history, tensor[0, :5])
    torch.testing.assert_close(future, tensor[0, 5:])


def test_dataset_rejects_wrong_window_length() -> None:
    tensor = torch.zeros(1, 6, 5, 1)

    with pytest.raises(ValueError, match="expected total_len=7"):
        WaveletSlidingWindowDataset(tensor, history_len=5, predict_len=2)
