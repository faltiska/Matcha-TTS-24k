from typing import Callable, Optional

import torch


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
    b = (backend or "vocos").lower()
    kwargs = dict(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
    )
    if b == "vocos":
        from matcha.vocos24k.mel_extractor import get_mel_extractor as _ext
        return _ext(**kwargs)
    if b == "bigvgan":
        from matcha.bigvgan24k.mel_extractor import get_mel_extractor as _ext
        return _ext(**kwargs)
    raise ValueError(f"Unknown mel backend: '{backend}'. Supported: 'vocos', 'bigvgan'")
