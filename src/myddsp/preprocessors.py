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
# TODO: fix mixed use of ndarray amd Tensor.


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


def make_divisible_by_hop_length(
    y: Tensor, window_length: int = C.N_FFT, hop_length: int = C.HOP_LENGTH
) -> Tensor:
    """Right pads a batch of audio examples to the nearest multiple of `HOP_SIZE`

    Args:
        y: tensor of shape `[..., S]`
        window_length: window length
        hop_length: hop length

    Returns:
        y_padded: right padded examples, tensor of shape `[..., S]`

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
    length = y.shape[-1]

    if length <= 0:
        raise ValueError("Example length must be strictly positive!")

    remainder = (length - window_length) % hop_length
    if remainder == 0:
        # already divisible by hop length
        return y
    padding = hop_length - remainder

    y_padded = F.pad(y, (0, padding), mode="constant", value=0.0)

    return y_padded


def center(y: Tensor, window_length: int = C.N_FFT, hop_length: int = C.HOP_LENGTH) -> Tensor:
    """Centers signal based on fft window size.

    Args:
        y: tensor of shape `[..., S]`
        window_length: window length
        hop_length: hop length

    Returns:
        y_centered: centered tensor of shape `[..., S]`

    Raises:
        ValueError: window length must be divisible by 2
    """
    if (window_length % 2) != 0:
        ValueError("Window length must be divisible by 2")

    y_divisible = make_divisible_by_hop_length(y, window_length, hop_length)
    padding = window_length // 2
    centered = F.pad(y_divisible, (padding, padding))

    return centered


def get_frames(
    y: np.array, window_length: int = C.N_FFT, hop_length: int = C.HOP_LENGTH
) -> np.array:
    """Generates frames of moving windows given hop length.

    Args:
        y: tensor of shape `[..., S]`
        window_length: length of each frame
        hop_length: hop length

    Returns:
        frames: audio frames of shape `[..., W, F]`

    Examples:
        >>> y = torch.randn(8, 2, 48000)
        >>> frames = get_frames(y)
        >>> frames.shape
        (8, 2, 3072, 235)

    Raises:
        ValueError: if tensor length is shorter than window length.
    """
    y_divisible = make_divisible_by_hop_length(y, window_length, hop_length)
    length = y_divisible.shape[-1]

    if length < window_length:
        raise ValueError("Example length cannot be shorter than the length of a single frame!")

    frames = librosa.util.frame(
        y_divisible, frame_length=window_length, hop_length=hop_length, writeable=True
    )

    return frames


def get_centered_frames(
    y: Tensor, window_length: int = C.N_FFT, hop_length: int = C.HOP_LENGTH
) -> Tensor:
    """Generates centered frames of moving windows given hop length.

    Centeres the audio before calling `get_frames` on it.

    Args:
        y: tensor of shape `[..., S]`
        window_length: length of each frame
        hop_length: hop length

    Returns:
        frames: audio frames of shape `[..., W, F]`

    Examples:
        >>> y = torch.randn(8, 2, 48000)
        >>> frames = get_centered_frames(y)
        >>> frames.shape
        (8, 2, 3072, 251)
    """
    centered = center(y, window_length, hop_length)
    frames = get_frames(centered, window_length, hop_length)

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
