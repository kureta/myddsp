import numpy as np
import torch
from hypothesis import Verbosity, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import array_shapes, arrays

from myddsp.preprocessors import get_centered_frames, get_frames

settings.register_profile("ci", max_examples=1000)
settings.register_profile("dev", max_examples=10)
settings.register_profile("debug", max_examples=10, verbosity=Verbosity.verbose)

upper = np.finfo("float32").max
lower = np.finfo("float32").min


@given(
    arrays(
        dtype=np.float32,
        shape=(8, 2, 48000),
        elements=st.floats(lower, upper, allow_nan=False, allow_infinity=False, width=32),
    )
)
def test_get_frames(data: np.array) -> None:
    y = torch.from_numpy(data)
    length = y.shape[-1]

    frames = get_frames(y, window_length=1024, hop_length=128)
    n_frames = frames.shape[-1]

    calc_n_frames = (length - 1024) // 128 + 1

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
