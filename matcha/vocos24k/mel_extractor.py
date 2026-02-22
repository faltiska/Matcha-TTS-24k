from typing import Callable

import torch
import torchaudio


def get_mel_extractor(
    *,
    sample_rate: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 100,
    log_eps: float = 1e-7,
    **_,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Mel extractor matching Vocos-24k training:
    - torchaudio MelSpectrogram with center=True, power=1 (magnitude), mel_scale='htk', norm=None
    - Audio trimmed to multiple of hop_length before STFT
    - Natural log with eps=1e-7
    """
    mel_spec = torchaudio.transforms.MelSpectrogram(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        center=True,
        power=1,
        mel_scale="htk",
        norm=None,
    )

    def extract_fn(y: torch.Tensor) -> torch.Tensor:
        device = y.device
        y = y[..., : (y.shape[-1] // hop_length) * hop_length]
        mel = mel_spec.to(device)(y)
        return torch.log(torch.clamp(mel, min=log_eps))

    return extract_fn
