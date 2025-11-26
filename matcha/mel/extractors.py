from typing import Callable, Optional

import torch
import torchaudio

# Reuse existing HiFi-GAN mel for the default backend
from matcha.utils.audio import mel_spectrogram as hifigan_mel


def _vocos_mel_factory(
    sample_rate: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 100,
    log_eps: float = 1e-7,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create a mel extractor matching Vocos-24k training:
    - torchaudio MelSpectrogram with center=True, power=1 (magnitude), mel_scale='htk', norm=None
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
        # mel_scale default is 'htk' for Vocos (see upstream); torchaudio default is 'htk' in recent versions
        mel_scale="htk",
        norm=None,
    )

    def extract_fn(y: torch.Tensor) -> torch.Tensor:
        # y: [B, T] or [1, T]
        device = y.device
        mel = mel_spec.to(device)(y)
        mel = torch.log(torch.clamp(mel, min=log_eps))
        return mel

    return extract_fn


def _hifigan_mel_factory(
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: Optional[float] = 8000.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Create a mel extractor matching current HiFi-GAN usage in this repository:
    - center=False
    - librosa mel basis via matcha.utils.audio.mel_spectrogram
    - Natural log compression with eps=1e-5 inside the helper
    """
    def extract_fn(y: torch.Tensor) -> torch.Tensor:
        return hifigan_mel(
            y,
            n_fft,
            n_mels,
            sample_rate,
            hop_length,
            win_length,
            f_min,
            f_max,
            center=False,
        )

    return extract_fn


def get_mel_extractor(
    backend: str,
    *,
    sample_rate: Optional[int] = None,
    n_fft: Optional[int] = None,
    hop_length: Optional[int] = None,
    win_length: Optional[int] = None,
    n_mels: Optional[int] = None,
    f_min: Optional[float] = None,
    f_max: Optional[float] = None,
) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Factory to obtain a mel spectrogram extractor callable consistent with the chosen backend.

    Args:
        backend: "hifigan" (default) or "vocos24k"
        Other params override sensible defaults for each backend if provided.

    Returns:
        Callable that maps waveform tensor [B, T] -> mel [B, n_mels, Frames]
    """
    b = (backend or "hifigan").lower()

    if b == "vocos":
        return _vocos_mel_factory(
            sample_rate=sample_rate or 24000,
            n_fft=n_fft or 1024,
            hop_length=hop_length or 256,
            win_length=win_length or 1024,
            n_mels=n_mels or 100,
            log_eps=1e-7,
        )
    # default: HiFi-GAN-style
    return _hifigan_mel_factory(
        sample_rate=sample_rate or 22050,
        n_fft=n_fft or 1024,
        hop_length=hop_length or 256,
        win_length=win_length or 1024,
        n_mels=n_mels or 80,
        f_min=(0.0 if f_min is None else f_min),
        f_max=(8000.0 if f_max is None else f_max),
    )
