import pickle
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import librosa
import numpy as np
import torch
import torchaudio.transforms
from rich.progress import Progress, track
from torch import Tensor, nn
from torch.nn import functional as F  # noqa
from torch.utils.data import Dataset

import myddsp.constants as C  # noqa
from myddsp.preprocessors import F0, Loudness, get_centered_frames

Z_DIM = 32
AMP_DIM = 128
PITCH_DIM = 360
STFT_DIM = 1536


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(STFT_DIM, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 64)
        self.fc6 = nn.Linear(64, Z_DIM)

    def forward(self, stft_frame: Tensor, *_: Tuple[Any]) -> Tensor:
        x = F.relu(self.fc1(stft_frame))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        z = self.fc6(x)

        return z


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(AMP_DIM + PITCH_DIM, 512)
        self.fc1_1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 1024)
        self.fc2_1 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 2048)
        self.fc3_1 = nn.Linear(2048, 2048)
        self.fc4 = nn.Linear(2048, 4096)
        self.fc4_1 = nn.Linear(4096, 4096)
        self.fc5 = nn.Linear(4096, STFT_DIM)

    def forward(self, amp: Tensor, pitch: Tensor) -> Tensor:
        cond = torch.cat([amp, pitch], dim=1)
        x = F.relu(self.fc1(cond))
        x = F.relu(self.fc1_1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc2_1(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc3_1(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc4_1(x))
        x = F.relu(self.fc5(x))

        return x


class VAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, stft_frame: Tensor, amp: Tensor, pitch: Tensor) -> Tensor:
        z = self.encoder(stft_frame, amp, pitch)
        x_hat = self.decoder(z, amp, pitch)

        return x_hat


def persist(file_name: str) -> Callable[..., Any]:
    functions = {
        ".npy": (lambda path, data: np.save(path, data), np.load),
        ".pt": (lambda path, data: torch.save(data, path), torch.load),
        ".pkl": (
            lambda path, data: pickle.dump(data, open(path, "wb")),
            lambda path: pickle.load(open(path, "rb")),
        ),
    }
    ext = Path(file_name).suffix
    try:
        save, load = functions[ext]
    except KeyError:
        raise ValueError(f"Extension {ext} is not supported")

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        def new_func(*args: Tuple[Any], **kwargs: Dict[str, Any]) -> Any:
            if not Path(file_name).exists():
                result = func(*args, **kwargs)
                save(file_name, result)
                return result
            else:
                return load(file_name)

        return new_func

    return decorator


def load_audio(path: Path) -> np.ndarray:
    y, _ = librosa.load(path, sr=C.SAMPLE_RATE, mono=True)
    return y


@persist("data/audio.pkl")
def load_file() -> List[np.ndarray]:
    path = Path("/home/kureta/Music/Cello Samples")
    files = list(path.glob("*.wav"))

    result = []
    with Progress() as progress:
        task = progress.add_task("Loading files...", total=len(files))
        with Pool(8) as pool:
            for y in pool.imap_unordered(load_audio, files):
                result.append(y)
                progress.update(task, advance=1)

    return result


@persist("data/stft.pt")
def prepare_stft() -> Tensor:
    # TODO: only first 500 files
    data = load_file()[:500]
    window = torch.windows.hann(C.N_FFT)
    specs: List[Tensor] = []
    for batch in track(data, description="Calculating STFT..."):
        frames = get_centered_frames(torch.from_numpy(batch)).squeeze().T
        stft = torch.abs(torch.fft.rfft(frames * window, norm="backward"))
        specs.append(stft)

    spec = torch.cat(specs)

    return spec[:, 1:]


@persist("data/amp.pt")
def prepare_amp() -> Tensor:
    # TODO: only first 500 files
    data = load_file()[:500]

    ld = Loudness()
    ld = ld.eval()
    for p in ld.parameters():
        p.requires_grad = False

    one_hot = []
    for batch in track(data, description="Calculating loudness..."):
        with torch.inference_mode():
            y = torch.from_numpy(batch).float()
            amps = ld(y.unsqueeze(0).unsqueeze(0)).flatten() / 100 + 1

        quantized = torch.round(amps * AMP_DIM).long()
        one_hot.append(torch.eye(AMP_DIM)[quantized])

    one_hot = torch.cat(one_hot)

    return one_hot


@persist("data/pitch.pt")
def prepare_pitch() -> Tensor:
    # TODO: only first 500 files
    data = load_file()[:500]

    crepe = F0("full").cuda()
    crepe = crepe.eval()
    for p in crepe.parameters():
        p.requires_grad = False

    rs = torchaudio.transforms.Resample(C.SAMPLE_RATE, C.CREPE_SAMPLE_RATE)
    pitch_conditions = []
    for batch in track(data, description="Calculating pitch..."):
        y = torch.from_numpy(batch).float()
        y = rs(y)
        y = y.unsqueeze(0).unsqueeze(0)
        # TODO: Processing entire file at once. Should use DataLoader instead.
        with torch.inference_mode():
            activations = crepe(y.cuda()).cpu()

        const = 9
        delta = 0.01
        peak_part = partial(
            librosa.util.peak_pick,
            pre_max=const,
            post_max=const,
            pre_avg=const,
            post_avg=const * 2,
            delta=delta,
            wait=const * 3,
        )
        result = list(map(peak_part, activations.cpu().numpy()))

        pitch_condition = np.zeros_like(activations.cpu())
        no_pitch = 0
        for idx, r in enumerate(result):
            try:
                pitch_condition[idx, r] = activations[idx, r].cpu()
            except IndexError:
                no_pitch += 1
                continue

        pitch_conditions.append(torch.from_numpy(pitch_condition).float())

    pitch_conditions = torch.cat(pitch_conditions)

    return pitch_conditions


def prepare_all() -> Tuple[Tensor, Tensor, Tensor]:
    stft = prepare_stft()
    amp = prepare_amp()
    pitch = prepare_pitch()
    return stft, amp, pitch


class ZakDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        self.stft, self.amp, self.pitch = prepare_all()

    def __len__(self) -> int:
        return int(self.stft.shape[0])

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        return self.stft[idx], self.amp[idx], self.pitch[idx]
