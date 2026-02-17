import torch
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, to_phoneme_ids, to_phonemes
from matcha.utils.utils import intersperse
from matcha.vocos24k.vocos_wrapper import load_model as load_vocos
from matcha.bigvgan24k.bigvgan_wrapper import load_bigvgan


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
