import torch
import torchaudio
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, to_phoneme_ids, to_phonemes
from matcha.utils.utils import intersperse
from matcha.vocos24k.vocos_wrapper import load_model as load_vocos
from matcha.bigvgan24k.bigvgan_wrapper import load_bigvgan

OUTPUT_SAMPLE_RATE = 16000
VOCODER_SAMPLE_RATE = 24000
VOCODER_HOP_LENGTH = 256
ODE_SOLVER = "midpoint"


def process_text(i: int, text: str, language: str, device: torch.device):
    print(f"[{i}] - Input text: {text}")
    phonemes = to_phonemes(text, language=language)
    phoneme_ids = to_phoneme_ids(phonemes)
    x = torch.tensor(
        intersperse(phoneme_ids, 0),
        dtype=torch.long,
        device=device,
    )[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    print(f"[{i}] - Phonetised text: {x_phones[1::2]}")

    return {"x_orig": text, "x": x, "x_lengths": x_lengths, "x_phones": x_phones}


def load_vocoder(vocoder_name, checkpoint_path, device):
    print(f"[!] Loading {vocoder_name}!")
    if vocoder_name == "vocos":
        vocoder = load_vocos(device)
        denoiser = None
    elif vocoder_name == "bigvgan":
        vocoder = load_bigvgan(device)
        denoiser = None
    else:
        raise NotImplementedError(f"Vocoder {vocoder_name} not implemented!")
    print(f"[+] {vocoder_name} loaded!")
    return vocoder, denoiser


def load_matcha(model_name, checkpoint_path, device):
    print(f"[!] Loading {model_name}!")
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device, weights_only=False, strict=False).eval()
    print(f"[+] {model_name} loaded!")
    return model


def to_waveform(mel, vocoder, denoiser=None, denoiser_strength=0.00025):
    audio = vocoder(mel)
    max_abs = audio.abs().max()
    if max_abs > 1.0:
        audio = audio / max_abs * 0.95
    return audio.cpu().squeeze()


def trim_trailing_silence(audio: torch.Tensor, sr: int, threshold_db: float = -60.0) -> torch.Tensor:
    threshold_amp = 10 ** (threshold_db / 20.0)
    window_frames = int(0.01 * sr)
    audio_squared = audio ** 2
    pad_size = window_frames - (len(audio) % window_frames)
    if pad_size < window_frames:
        audio_squared = torch.nn.functional.pad(audio_squared, (0, pad_size))
    audio_squared = audio_squared.reshape(-1, window_frames)
    rms = torch.sqrt(audio_squared.mean(dim=1))
    for i in range(len(rms) - 1, -1, -1):
        if rms[i] >= threshold_amp:
            return audio[:(i + 1) * window_frames]
    return audio


def apply_lowpass_filter(audio: torch.Tensor, sr: int, start_freq: float = 7000, end_freq: float = 8000, end_db: float = -15.0) -> torch.Tensor:
    fft = torch.fft.rfft(audio)
    freqs = torch.fft.rfftfreq(len(audio), 1/sr)
    gain = torch.ones_like(freqs)
    mask = (freqs >= start_freq) & (freqs <= end_freq)
    gain[mask] = torch.pow(10, (end_db / 20.0) * (freqs[mask] - start_freq) / (end_freq - start_freq))
    gain[freqs > end_freq] = 10 ** (end_db / 20.0)
    return torch.fft.irfft(fft * gain, n=len(audio))


def post_process(audio, orig_freq=VOCODER_SAMPLE_RATE, target_freq=OUTPUT_SAMPLE_RATE):
    import time
    start = time.perf_counter()
    audio = trim_trailing_silence(audio, orig_freq)
    audio = torchaudio.functional.resample(audio, orig_freq=orig_freq, new_freq=target_freq)
    audio = apply_lowpass_filter(audio, target_freq)
    print(f"Post-processing took {time.perf_counter() - start:.3f}s")
    return audio
