"""
MCD validation script variant with 24 kHz and 30 coefficients.
The original was using a python library that converted the audio to 22KHz and used 13 coefficients only.
The tradition of using 13 is a relic from Automatic Speech Recognition (ASR) working on 8 kHz or 16 kHz telephone and 
broadcast audio, which typically used only 20 to 40 mel bins.
That is not enough for myu model. It was like applying a low pass to the audio.
In modern speech synthesis literature (evaluating Tacotron, FastSpeech, VITS, and HiFi-GAN), researchers routinely use
24 to 35 Mel-Generalized Cepstral (MGC) coefficients or MFCCs when evaluating 22.05 kHz or 24 kHz audio.

Identical to mcd_validate.py, but uses a subclass of Calculate_MCD that overrides:
  - SAMPLING_RATE: 24000 Hz  (matches the corpus and the model, avoids resampling to 22050)
  - order:         30        (captures the full perceptually relevant cepstral range, vs the default 13)
  - alpha:         0.66      (frequency warping coefficient tuned for 24 kHz, vs the default 0.65 for 22050 Hz)

MCD values from this script are NOT directly comparable to mcd_validate.py values,
because a different order and sample rate produce a different cepstral scale.
Use each script consistently within its own measurement series.

Usage:
    python -m matcha.utils.mcd_validate_v2 --checkpoint your.ckpt
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyworld")

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pyworld
import pysptk
import soundfile as sf
import torch
import torchaudio as ta
from pymcd.mcd import Calculate_MCD

from matcha.inference import VOICES, DEFAULT_ODE_SOLVER, DEFAULT_NUM_STEPS, load_matcha, load_vocoder, pipeline
from matcha.utils.precompute_mels import _load_yaml_config, _resolve_path, parse_filelist

SAMPLE_OFFSET = 10
SAMPLES_PER_SPEAKER = 20


class Calculate_MCD_24k(Calculate_MCD):
    """
    Calculate_MCD subclass that runs analysis at 24 kHz with 30 cepstral coefficients.

    The default Calculate_MCD hardcodes:
      - SAMPLING_RATE = 22050  (resamples all audio before analysis)
      - order = 13             (MCD convention, historical)
      - alpha = 0.65           (frequency warping for 22050 Hz)

    This subclass overrides all three to match the corpus sample rate and use a
    perceptually richer cepstral representation consistent with the cepstral training loss.
    """

    SAMPLING_RATE_24K = 24000
    ORDER = 30
    # Frequency warping coefficient for 24 kHz. The standard value for 22050 Hz is 0.65;
    # for 24 kHz the correct value is 0.66.
    ALPHA_24K = 0.66

    def __init__(self, MCD_mode):
        super().__init__(MCD_mode)
        self.SAMPLING_RATE = self.SAMPLING_RATE_24K

    def wav2mcep_numpy(self, loaded_wav, fft_size=512):
        _, sp, _ = pyworld.wav2world(loaded_wav.astype(np.double), fs=self.SAMPLING_RATE,
                                     frame_period=self.FRAME_PERIOD, fft_size=fft_size)
        return pysptk.sptk.mcep(sp, order=self.ORDER, alpha=self.ALPHA_24K, maxiter=0,
                                 etype=1, eps=1.0E-8, min_det=0.0, itype=3)


def pick_samples(valid_filelist: Path, speaker_id: str) -> list[tuple[str, Path]]:
    """Return SAMPLES_PER_SPEAKER rows starting at SAMPLE_OFFSET from validate.csv for the given speaker."""
    all_rows = parse_filelist(valid_filelist)
    samples = []
    skipped = 0
    for rel_path, spk_id, _lang, text, _phoneme_ids in all_rows:
        if spk_id != speaker_id:
            continue
        if skipped < SAMPLE_OFFSET:
            skipped += 1
            continue
        wav_path = (valid_filelist.parent / "wav" / (rel_path + ".wav")).resolve()
        samples.append((text, wav_path))
        if len(samples) == SAMPLES_PER_SPEAKER:
            break
    return samples


def trim_silence(audio: torch.Tensor, sr: int, threshold_db: float = -60.0) -> torch.Tensor:
    threshold_amp = 10 ** (threshold_db / 20.0)
    window_frames = int(0.01 * sr)
    pad_size = window_frames - (len(audio) % window_frames)
    if pad_size < window_frames:
        audio = torch.nn.functional.pad(audio, (0, pad_size))
    rms = audio.reshape(-1, window_frames).pow(2).mean(dim=1).sqrt()
    start = next((i for i, r in enumerate(rms) if r >= threshold_amp), 0)
    end = next((i for i in range(len(rms) - 1, -1, -1) if rms[i] >= threshold_amp), len(rms) - 1) + 1
    return audio[start * window_frames : end * window_frames]


def compute_mcd(gen_wav: torch.Tensor, ref_wav_path: Path, mcd_toolbox: Calculate_MCD, sample_rate: int) -> tuple[float, float]:
    """Returns (mcd, gt_duration / infer_duration) after trimming silence from both."""
    gen_trimmed = trim_silence(gen_wav.cpu(), sample_rate)
    ref_audio, ref_sr = ta.load(str(ref_wav_path))
    ref_trimmed = trim_silence(ref_audio.squeeze(0), ref_sr)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as gen_f, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_f:
        gen_path, ref_path = gen_f.name, ref_f.name

    sf.write(gen_path, gen_trimmed.numpy(), sample_rate)
    sf.write(ref_path, ref_trimmed.numpy(), ref_sr)

    mcd = mcd_toolbox.calculate_mcd(gen_path, ref_path)
    Path(gen_path).unlink(missing_ok=True)
    Path(ref_path).unlink(missing_ok=True)

    gt_duration = len(ref_trimmed) / ref_sr
    infer_duration = len(gen_trimmed) / sample_rate
    duration_ratio = gt_duration / infer_duration if infer_duration > 0 else 1.0
    return mcd, duration_ratio


def main():
    parser = argparse.ArgumentParser(description="MCD validation (24 kHz, 30 coefficients): compare TTS output to ground truth per speaker")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data-config", default="configs/data/corpus-24k.yaml")
    parser.add_argument("--vocoder", default="vocos", choices=["vocos"])
    parser.add_argument("--solver", type=str, default=DEFAULT_ODE_SOLVER)
    parser.add_argument("--steps", type=int, default=DEFAULT_NUM_STEPS)
    args = parser.parse_args()

    ckpt_name = Path(args.checkpoint).stem
    print(f"Processing {ckpt_name}...")

    cfg = _load_yaml_config(Path(args.data_config).resolve())
    valid_filelist = _resolve_path(str(cfg["valid_filelist_path"]))
    sample_rate = int(cfg["sample_rate"])

    model = load_matcha("custom_model", args.checkpoint)
    vocoder = load_vocoder(args.vocoder)
    mcd_toolbox = Calculate_MCD_24k(MCD_mode="dtw")

    for voice in VOICES:
        voice["scale_correction"] = 1.0

    speaker_mcd_scores: dict[str, tuple[float, float]] = {}  # voice_id -> (avg_mcd, avg_duration_ratio)

    for voice in VOICES:
        spk_id = int(voice["id"])
        language = voice["lang"]
        samples = pick_samples(valid_filelist, voice["id"])
        if not samples:
            print(f"[!] No samples found for speaker {spk_id}, skipping.")
            continue

        mcd_scores = []
        duration_ratios = []
        for text, gt_wav_path in samples:
            waveform = pipeline(model, vocoder, text, spk_id, None, args.steps)
            mcd, duration_ratio = compute_mcd(waveform, gt_wav_path, mcd_toolbox, sample_rate)
            mcd_scores.append(mcd)
            duration_ratios.append(duration_ratio)
        speaker_mcd_scores[voice["id"]] = (sum(mcd_scores) / len(mcd_scores), sum(duration_ratios) / len(duration_ratios))

    print()
    for voice_id, (avg_mcd, avg_ratio) in speaker_mcd_scores.items():
        label = f"speaker_{int(voice_id):03d}"
        print(f"{label:<40} MCD: {avg_mcd:5.2f} dB   duration ratio: {avg_ratio:.2f}")
    print("-" * 70)
    all_mcds = [mcd for mcd, _ in speaker_mcd_scores.values()]
    if all_mcds:
        print(f"{'Average':<40} MCD: {sum(all_mcds) / len(all_mcds):5.2f} dB")
    print()
    print(f"Completed {ckpt_name}")


if __name__ == "__main__":
    main()
