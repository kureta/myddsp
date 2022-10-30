"""Provides a few audio feature extractors.

This module allows the user to extract audio features for feeding into the model.

The module contains the following feature extractors:

- `LegacyLoudness` - The one I used originally. Its dynamic range is somehow compressed, and it is noisy.
- `Loudness` - Based on`Loudness` from `torchaudio`.
"""

import librosa
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F

from .constants import HOP_LENGTH, N_FFT, SAMPLE_RATE


def phase_shuffle(y: Tensor) -> Tensor:
    """Randomizes the phase of an audio signal.

    Args:
        y: tensor of shape `[..., S]`

    Returns:
        y_phase_randomized: phase shuffled tensor of shape `[..., S]`
    """
    s = torch.fft.rfft(y)
    s_phase_randomized = torch.polar(s.abs(), torch.rand_like(s.angle()) * 2.0 * np.pi)
    y_phase_randomized = torch.fft.irfft(s_phase_randomized)

    return y_phase_randomized


def make_divisible_by_hop_length(y: Tensor) -> Tensor:
    """Right pads a batch of audio examples to the nearest multiple of `HOP_SIZE`

    !!! warning
        Example length cannot be zero.

    Examples:
        >>> y = torch.randn(1, 2, 48003)
        >>> y_padded = make_divisible_by_hop_length(y)
        >>> y_padded.shape[-1] % HOP_LENGTH == 0
        True

        >>> y = torch.randn(1, 2, 0)
        >>> make_divisible_by_hop_length(y)
        Traceback (most recent call last):
            ...
        ValueError:...

    Args:
        y: batch of examples, tensor of shape `[B, C, S]`

    Returns:
        y_padded: padded examples, tensor of shape `[B, C, S]`
    """
    if len(y.shape) != 3:
        raise ValueError("Tensor must have exactly 3 dimensions [B, C, S]")

    length = y.shape[2]

    if length <= 0:
        raise ValueError("Example length cannot be zero!")

    remainder = length % HOP_LENGTH
    padding = HOP_LENGTH - remainder

    y_padded = F.pad(y, (0, padding), mode="constant", value=0.0)

    return y_padded


class LegacyLoudness(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        frequencies = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT).astype("float32")
        a_weighting = librosa.A_weighting(frequencies)[None, :].astype("float32")
        self.register_buffer("a_weighting", torch.from_numpy(a_weighting))

    def forward(self, x: Tensor) -> Tensor:
        # to mono
        x = x.mean(1)
        window = torch.hann_window(N_FFT).to(x.device)
        s = torch.stft(
            x,
            n_fft=N_FFT,
            hop_length=HOP_LENGTH,
            window=window,
            return_complex=True,
            # pad_mode="reflect",
            center=False,
            normalized=True,
        ).transpose(1, 2)

        # Compute power.
        amplitude = torch.abs(s)
        power = amplitude**2

        weighting = 10 ** (self.a_weighting / 10)
        power = power * weighting

        power = torch.mean(power, dim=-1)
        loudness = 10.0 * torch.log10(
            torch.maximum(torch.tensor(1e-10, device=power.device), power)
        )
        loudness = torch.maximum(loudness, loudness.max() - 80.0)

        return loudness.unsqueeze(1)
