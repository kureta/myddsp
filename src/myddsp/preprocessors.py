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

import myddsp.constants as C

# TODO: use the same loudness to trim silence.


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

    Args:
        y: batch of examples, tensor of shape `[B, C, S]`

    Returns:
        y_padded: padded examples, tensor of shape `[B, C, S]`

    Raises:
        ValueError: if tensor is not in the required shape.

    Examples:
        >>> y = torch.randn(1, 2, 48003)
        >>> y_padded = make_divisible_by_hop_length(y)
        >>> y_padded.shape[-1] % C.HOP_LENGTH == 0
        True

        >>> y = torch.randn(1, 2, 0)
        >>> make_divisible_by_hop_length(y)
        Traceback (most recent call last):
            ...
        ValueError:...
    """
    if len(y.shape) != 3:
        raise ValueError("Tensor must have exactly 3 dimensions [B, C, S]")

    length = y.shape[2]

    if length <= 0:
        raise ValueError("Example length cannot be zero!")

    remainder = length % C.HOP_LENGTH
    padding = C.HOP_LENGTH - remainder

    y_padded = F.pad(y, (0, padding), mode="constant", value=0.0)

    return y_padded


def get_frames(y: np.array, hop_length: int = C.HOP_LENGTH) -> np.array:
    """Generates frames of moving windows given hop length.

    Window size is a constant, hop length may become a parameter in the future.

    Args:
        y: tensor of shape `[..., S]`
        hop_length: hop length

    Returns:
        frames: audio frames of shape `[..., W, F]`

    Examples:
        >>> y = np.random.randn(8, 2, 48000)
        >>> frames = get_frames(y)
        >>> frames.shape
        (8, 2, 3072, 235)

    Raises:
        ValueError: if tensor length is not divisible by hop length or shorter than window length.
    """
    length = y.shape[-1]
    if (length % hop_length) != 0:
        raise ValueError("Example length must be divisible by hop length!")

    if length < C.N_FFT:
        raise ValueError("Example length cannot be shorter than the length of a single frame!")

    frames = librosa.util.frame(y, frame_length=C.N_FFT, hop_length=hop_length, writeable=True)

    return frames


class LegacyLoudness(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        frequencies = librosa.fft_frequencies(sr=C.SAMPLE_RATE, n_fft=C.N_FFT).astype("float32")
        a_weighting = librosa.A_weighting(frequencies)[None, :].astype("float32")
        self.register_buffer("a_weighting", torch.from_numpy(a_weighting))

    def forward(self, x: Tensor) -> Tensor:
        # to mono
        x = x.mean(1)
        window = torch.hann_window(C.N_FFT).to(x.device)
        s = torch.stft(
            x,
            n_fft=C.N_FFT,
            hop_length=C.HOP_LENGTH,
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
