"""HiFiGAN-16 kHz vocoder wrapper for Matcha-TTS."""

import torch
import torch.nn as nn
import torchaudio

# -----------------------------------------------------------------------------
# 1)  Torchaudio stub ─ SpeechBrain expects torchaudio.list_audio_backends().
#     The custom 2.9.1 wheel you use does not expose that symbol, so we inject
#     a harmless stub if it is missing.
# -----------------------------------------------------------------------------
if not hasattr(torchaudio, "list_audio_backends"):
    torchaudio.list_audio_backends = lambda: []  # type: ignore[assignment]

# -----------------------------------------------------------------------------
# 2)  Hugging Face shim ─ Older SpeechBrain versions still call
#     huggingface_hub.hf_hub_download(..., use_auth_token=...).
#     Recent huggingface-hub wheels removed that kwarg.  We wrap the function so
#     it tolerates the obsolete parameter.
# -----------------------------------------------------------------------------
import inspect
import huggingface_hub

if "use_auth_token" not in inspect.signature(huggingface_hub.hf_hub_download).parameters:
    _orig_hf_hub_download = huggingface_hub.hf_hub_download  # type: ignore

    def _hf_hub_download_with_token(*args, use_auth_token=None, **kwargs):  # noqa: D401
        # Simply ignore the deprecated argument and forward the call.
        return _orig_hf_hub_download(*args, **kwargs)

    huggingface_hub.hf_hub_download = _hf_hub_download_with_token  # type: ignore

# -----------------------------------------------------------------------------
# 3)  Import SpeechBrain HiFiGAN after the monkey-patches so its import-time
#     checks succeed.
# -----------------------------------------------------------------------------
from speechbrain.inference.vocoders import HIFIGAN


class SpeechBrainHiFiGANWrapper(nn.Module):
    """
    Thin wrapper so the SpeechBrain HiFiGAN matches the interface used by
    Matcha-TTS (forward(mel) ➜ waveform).
    """

    def __init__(self, model: "HIFIGAN"):
        super().__init__()
        self.model = model

    def forward(self, mel: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Decode a batch of mel-spectrograms to waveforms.

        Ensures the mel tensor is moved to the same device as the model to avoid
        CPU↔CUDA mismatches (FloatTensor vs. CUDAFloatTensor).
        """
        device = next(self.model.parameters()).device
        mel = mel.to(device, non_blocking=True)
        return self.model.decode_batch(mel)


def load_vocoder(
    model_id: str = "speechbrain/tts-hifigan-libritts-16kHz",
    cache_dir: str = "pretrained_models/tts-hifigan-libritts-16kHz",
    device: str = "cuda",
) -> SpeechBrainHiFiGANWrapper:
    """
    Download & load the SpeechBrain 16 kHz HiFiGAN vocoder without 404 errors.

    SpeechBrain’s helper `HIFIGAN.from_hparams(source, savedir)` currently
    points to a sub-folder that no longer exists on the Hub.  We follow the
    pattern used in *istftnet/wrapper.py*:

        1. snapshot the repository locally with
           ``huggingface_hub.snapshot_download``
        2. let ``HIFIGAN.from_hparams`` load weights from that **local** path.

    Parameters
    ----------
    model_id :
        Hugging Face repo id (default ``speechbrain/tts-hifigan-libritts-16kHz``).
    cache_dir :
        Local directory used to cache / symlink the snapshot.
    device :
        ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    SpeechBrainHiFiGANWrapper
        Ready-to-use vocoder.
    """
    from huggingface_hub import snapshot_download

    # Step-1: ensure repo is present locally (downloads only missing files).
    local_dir = snapshot_download(
        repo_id=model_id,
        cache_dir=cache_dir,
        local_files_only=False,
    )

    # Step-2: load the model weights from the local snapshot.
    model = HIFIGAN.from_hparams(source=local_dir)
    model = model.eval().to(device)
    # SpeechBrain keeps an internal ``self.device`` attribute that is **not**
    # updated by .to(device).  If it still reads "cpu", the inference helper
    # will move the spectrogram back to CPU and trigger a dtype/device mismatch.
    # We overwrite it so that ``self.infer(spectrogram.to(self.device))`` picks
    # the actual device the weights live on.
    import torch as _torch

    model.device = _torch.device(device) if isinstance(device, str) else device

    return SpeechBrainHiFiGANWrapper(model)
