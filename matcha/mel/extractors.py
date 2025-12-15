from typing import Callable, Optional

import torch
import torchaudio

# Reuse existing HiFi-GAN mel for the default backend
from matcha.utils.audio import mel_spectrogram as hifigan_mel


class VocosMelExtractor:
    """Introduced a class, to avoid pickling errors on Windows."""
    def __init__(
        self,
        sample_rate: int = 24000,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 100,
        log_eps: float = 1e-7,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.log_eps = log_eps
        self.mel_spec = None
    
    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        if self.mel_spec is None:
            self.mel_spec = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.win_length,
                n_mels=self.n_mels,
                center=True,
                power=1,
                mel_scale="htk",
                norm=None,
            )
        
        device = y.device
        mel = self.mel_spec.to(device)(y)
        mel = torch.log(torch.clamp(mel, min=self.log_eps))
        return mel


class HifiganMelExtractor:
    """Picklable HiFiGAN mel extractor class"""
    def __init__(
        self,
        sample_rate: int = 22050,
        n_fft: int = 1024,
        hop_length: int = 256,
        win_length: int = 1024,
        n_mels: int = 80,
        f_min: float = 0.0,
        f_max: Optional[float] = 8000.0,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
    
    def __call__(self, y: torch.Tensor) -> torch.Tensor:
        return hifigan_mel(
            y,
            self.n_fft,
            self.n_mels,
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.f_min,
            self.f_max,
            center=False,
        )


def _vocos_mel_factory(
    sample_rate: int = 24000,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 100,
    log_eps: float = 1e-7,
) -> Callable[[torch.Tensor], torch.Tensor]:
    return VocosMelExtractor(sample_rate, n_fft, hop_length, win_length, n_mels, log_eps)


def _hifigan_mel_factory(
    sample_rate: int = 22050,
    n_fft: int = 1024,
    hop_length: int = 256,
    win_length: int = 1024,
    n_mels: int = 80,
    f_min: float = 0.0,
    f_max: Optional[float] = 8000.0,
) -> Callable[[torch.Tensor], torch.Tensor]:
    return HifiganMelExtractor(sample_rate, n_fft, hop_length, win_length, n_mels, f_min, f_max)


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
