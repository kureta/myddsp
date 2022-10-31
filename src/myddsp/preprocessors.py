"""Provides a few audio feature extractors.

This module allows the user to extract audio features for feeding into the model.

The module contains the following feature extractors:

- `LegacyLoudness` - The one I used originally. Its dynamic range is somehow compressed, and it is noisy.
- `Loudness` - Based on`Loudness` from `torchaudio`.
"""
import einops
import librosa
import torch
from torch import Tensor, nn
from torch.nn import functional as F  # noqa

import myddsp.constants as C  # noqa
from myddsp.crepe import load_model

# TODO: use the same loudness to trim silence.


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
        >>> signal = torch.randn(1, 2, 48003)
        >>> signal_padded = make_divisible_by_hop_length(signal)
        >>> signal_padded.shape[-1] % C.HOP_LENGTH == 0
        True

        >>> signal = torch.randn(1, 2, 0)
        >>> make_divisible_by_hop_length(signal)
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


def make_frames(y: Tensor, window_length: int = C.N_FFT, hop_length: int = C.HOP_LENGTH) -> Tensor:
    return y.unfold(-1, window_length, hop_length).transpose(-1, -2)


def get_frames(y: Tensor, window_length: int = C.N_FFT, hop_length: int = C.HOP_LENGTH) -> Tensor:
    """Generates frames of moving windows given hop length.

    Args:
        y: tensor of shape `[..., S]`
        window_length: length of each frame
        hop_length: hop length

    Returns:
        frames: audio frames of shape `[..., W, F]`

    Examples:
        >>> signal = torch.randn(8, 2, 48000)
        >>> signal_frames = get_frames(signal)
        >>> signal_frames.shape
        torch.Size([8, 2, 3072, 235])

    Raises:
        ValueError: if tensor length is shorter than window length.
    """
    y_divisible = make_divisible_by_hop_length(y, window_length, hop_length)
    length = y_divisible.shape[-1]

    if length < window_length:
        raise ValueError("Example length cannot be shorter than the length of a single frame!")

    frames = make_frames(y_divisible, window_length, hop_length)

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
        >>> signal = torch.randn(8, 2, 48000)
        >>> signal_frames = get_centered_frames(signal)
        >>> signal_frames.shape
        torch.Size([8, 2, 3072, 251])
    """
    centered = center(y, window_length, hop_length)
    frames = get_frames(centered, window_length, hop_length)

    return frames


class Loudness(nn.Module):
    def __init__(self, window_length: int = C.N_FFT):
        super().__init__()
        self.window_length = window_length

        frequencies = librosa.fft_frequencies(sr=C.SAMPLE_RATE, n_fft=C.N_FFT).astype("float32")
        a_weighting = librosa.A_weighting(frequencies).astype("float32")
        a_weighting = 10 ** (a_weighting / 10)
        self.register_buffer("a_weighting", torch.from_numpy(a_weighting))

        window = torch.hann_window(window_length)
        self.register_buffer("window", window)

    def forward(self, x: Tensor) -> Tensor:
        x = x.mean(1)
        frames = get_centered_frames(x)

        windowed = einops.einsum(frames, self.window, "b n f, n -> b n f")

        amplitude = torch.fft.rfft(windowed, dim=-2, norm="ortho").abs()

        power = amplitude**2
        power = einops.einsum(power, self.a_weighting, "b n f, n -> b n f")

        power = torch.mean(power, dim=-2, keepdim=True)
        loudness = 10.0 * torch.log10(
            torch.maximum(torch.tensor(1e-10, device=power.device), power)
        )
        loudness = torch.maximum(loudness, loudness.max() - 80.0)

        return loudness


class F0(nn.Module):
    def __init__(self, capacity: str = "full"):
        super().__init__()
        self.model = load_model(capacity)

    def forward(self, x: Tensor) -> Tensor:
        x = x.mean(1)
        frames = get_centered_frames(x, C.CREPE_N_FFT, C.CREPE_HOP_LENGTH)

        batched = einops.rearrange(frames, "b n f -> (b f) n")
        zeroed = batched - batched.mean(dim=1, keepdim=True)
        normalized = zeroed / zeroed.std(dim=1, keepdim=True)

        activations = self.model(normalized)

        return activations
