import librosa
import torch
import torch.nn as nn

from .constants import HOP_LENGTH, N_FFT, SAMPLE_RATE


class Loudness(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        frequencies = librosa.fft_frequencies(sr=SAMPLE_RATE, n_fft=N_FFT).astype("float32")
        a_weighting = librosa.A_weighting(frequencies)[None, :].astype("float32")
        self.register_buffer("a_weighting", torch.from_numpy(a_weighting))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
