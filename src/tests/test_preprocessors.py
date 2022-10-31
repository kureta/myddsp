import torch

from myddsp.preprocessors import get_centered_frames, get_frames


def test_get_frames() -> None:
    y = torch.randn(8, 2, 48000)
    frames = get_frames(y, window_length=1024, hop_length=128)

    n_frames = frames.shape[-1]
    calc_n_frames = (48000 - 1024) // 128 + 1

    assert n_frames == calc_n_frames


def test_get_centered_frames() -> None:
    y = torch.randn(8, 2, 48000)
    frames = get_centered_frames(y, window_length=1024, hop_length=128)

    n_frames = frames.shape[-1]
    calc_n_frames = 48000 // 128 + 1

    assert n_frames == calc_n_frames


def test_get_centered_frames2() -> None:
    y = torch.randn(48000)
    frames = get_centered_frames(y, window_length=1024, hop_length=128)

    n_frames = frames.shape[-1]
    calc_n_frames = 48000 // 128 + 1

    assert n_frames == calc_n_frames
