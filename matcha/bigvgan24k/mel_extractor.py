from typing import Callable

import torch

from matcha.bigvgan24k.meldataset import mel_spectrogram


def get_mel_extractor(
    *,
    sample_rate: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 100,
    f_min: float = 0.0,
    f_max: float = None,
    **_,
) -> Callable[[torch.Tensor], torch.Tensor]:
    def extract_fn(y: torch.Tensor) -> torch.Tensor:
        y = y[..., : (y.shape[-1] // hop_length) * hop_length]
        return mel_spectrogram(y, n_fft, n_mels, sample_rate, hop_length, win_length, f_min, f_max)

    return extract_fn
