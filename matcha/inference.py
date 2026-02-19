import io
import time
import lameenc
import torch
import torchaudio
from matcha.models.matcha_tts import MatchaTTS
from matcha.text import sequence_to_text, to_phoneme_ids, to_phonemes
from matcha.utils.utils import intersperse
from matcha.vocos24k.vocos_wrapper import load_model as load_vocos
from matcha.bigvgan24k.bigvgan_wrapper import load_bigvgan
import av
import numpy as np

SAMPLE_RATE = 24000
HOP_LENGTH = 256
ODE_SOLVER = "midpoint"


def process_text(text: str, language: str, device: torch.device):
    print(f"Input text: {text}")
    phonemes = to_phonemes(text, language=language)
    phoneme_ids = to_phoneme_ids(phonemes)
    x = torch.tensor(
        intersperse(phoneme_ids, 0),
        dtype=torch.long,
        device=device,
    )[None]
    x_lengths = torch.tensor([x.shape[-1]], dtype=torch.long, device=device)
    x_phones = sequence_to_text(x.squeeze(0).tolist())
    print(f"Phonetised text: {x_phones[1::2]}")

    return {"x_orig": text, "x": x, "x_lengths": x_lengths, "x_phones": x_phones}


def load_vocoder(vocoder_name, device):
    print(f"[!] Loading {vocoder_name}!")
    if vocoder_name == "vocos":
        vocoder = load_vocos(device)
    elif vocoder_name == "bigvgan":
        vocoder = load_bigvgan(device)
    else:
        raise NotImplementedError(f"Vocoder {vocoder_name} not implemented!")
    print(f"[+] {vocoder_name} loaded!")
    return vocoder


def load_matcha(model_name, checkpoint_path, device):
    print(f"[!] Loading {model_name}!")
    model = MatchaTTS.load_from_checkpoint(checkpoint_path, map_location=device, weights_only=False, strict=False).eval()
    return model


def synthesise(model, vocoder, text, language, spk=0, voice_mix=None, n_timesteps=15, length_scale=1.0):
    text_processed = process_text(text, language, next(model.parameters()).device)
    output = model.synthesise(
        text_processed["x"],
        text_processed["x_lengths"],
        n_timesteps=n_timesteps,
        spks=spk,
        voice_mix=voice_mix,
        length_scale=length_scale,
    )
    waveform = to_waveform(output["mel"], vocoder)
    return post_process(waveform), output["rtf"]


def to_waveform(mel, vocoder):
    audio = vocoder(mel)
    max_abs = audio.abs().max()
    if max_abs > 1.0:
        audio = audio / max_abs * 0.95
    return audio.cpu().squeeze()


def apply_lowpass_filter(audio: torch.Tensor, start_freq: float = 11000, end_freq: float = 12000, end_db: float = -12.0) -> torch.Tensor:
    fft = torch.fft.rfft(audio)
    freqs = torch.fft.rfftfreq(len(audio), 1/SAMPLE_RATE)
    gain = torch.ones_like(freqs)
    mask = (freqs >= start_freq) & (freqs <= end_freq)
    gain[mask] = torch.pow(10, (end_db / 20.0) * (freqs[mask] - start_freq) / (end_freq - start_freq))
    gain[freqs > end_freq] = 10 ** (end_db / 20.0)
    return torch.fft.irfft(fft * gain, n=len(audio))


def post_process(audio):
    start = time.perf_counter()
    audio = apply_lowpass_filter(audio)
    print(f"Post-processing took {time.perf_counter() - start:.3f}s")
    return audio


def convert_to_mp3(waveform):
    start = time.perf_counter()
    waveform_int16 = (waveform * 32767).to(torch.int16)
    wav_size = waveform_int16.numel() * 2
    
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(128)
    encoder.set_in_sample_rate(SAMPLE_RATE)
    encoder.set_channels(1)
    encoder.set_quality(5)
    
    mp3_data = encoder.encode(waveform_int16.numpy().tobytes())
    mp3_data += encoder.flush()
    
    mp3_size = len(mp3_data)
    pct = (mp3_size / wav_size * 100) if wav_size > 0 else 0
    print(f"MP3 conversion: {(time.perf_counter() - start)*1000:.1f}ms | {pct:.0f}% size")
    return bytes(mp3_data)


def convert_to_opus_ogg(waveform):
    start = time.perf_counter()
    audio_np = (waveform.numpy() * 32767).astype(np.int16).reshape(1, -1)
    wav_size = audio_np.size * 2
    
    buffer = io.BytesIO()
    container = av.open(buffer, mode='w', format='ogg')
    stream = container.add_stream('libopus', rate=SAMPLE_RATE)
    stream.layout = 'mono'
    stream.bit_rate = 48000
    stream.options = {'compression_level': "5"}
    
    frame = av.AudioFrame.from_ndarray(audio_np, format='s16', layout='mono')
    frame.sample_rate = SAMPLE_RATE
    
    for packet in stream.encode(frame):
        container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    
    container.close()
    
    ogg_data = buffer.getvalue()
    ogg_size = len(ogg_data)
    pct = (ogg_size / wav_size * 100) if wav_size > 0 else 0
    print(f"OGG conversion: {(time.perf_counter() - start)*1000:.1f}ms | {pct:.0f}% size")
    return bytes(ogg_data)
