"""
Combined validation script: MCD (24 kHz, 30 coefficients) + UTMOS, sharing a single inference pass.

Running mcd_validate_v2.py and utmos_validate.py separately means loading the model twice and
running inference over all samples twice. Since both scripts pick the same samples per speaker
and generate the same waveform via the same pipeline() call, this script generates each waveform
once and feeds it to both scorers.

MCD (reference-based, lower is better):
  Identical to mcd_validate_v2.py. Uses a subclass of Calculate_MCD that overrides:
    - SAMPLING_RATE: 24000 Hz  (matches the corpus and the model, avoids resampling to 22050)
    - order:         30        (captures the full perceptually relevant cepstral range, vs the default 13)
    - alpha:         0.66      (frequency warping coefficient tuned for 24 kHz, vs the default 0.65 for 22050 Hz)
  MCD values from this script are NOT directly comparable to mcd_validate.py (the 13-coefficient,
  22050 Hz variant) values, because a different order and sample rate produce a different cepstral scale.

UTMOS (reference-free, higher is better, 1-5 scale):
  https://arxiv.org/abs/2204.02152
  https://github.com/sarulab-speech/UTMOS22
  Predicted naturalness score. Production-quality TTS usually lands above 4.0. The model is
  downloaded on first run via torch.hub from tarepan/SpeechMOS and cached under ~/.cache/torch/hub.

  UTMOSv2 (SLT 2024) was tried and rejected: it costs about 1 s per utterance against v1's ~0.05 s,
  roughly 7x the total runtime, because it runs four 512x512 mel images through EfficientNetV2-S
  alongside wav2vec2 instead of a single wav2vec2 pass. On the v23 epoch 1014 checkpoint it ranked
  all 15 speakers in the same order as v1, so the extra cost bought nothing for checkpoint tracking.
  Batching to recover the time does not work either: predict() needs equal-length input, and the
  zero padding shifts scores by up to 0.76 MOS. v2's strength is ranking different systems against
  each other (VoiceMOS 2024), which is not what this script does.

Both scorers pick SAMPLES_PER_SPEAKER samples per speaker, starting at SAMPLE_OFFSET, from validate.csv,
forcing scale_correction to 1.0 for every voice before running.

Usage:
    python -m matcha.utils.validate --checkpoint your.ckpt
    python -m matcha.utils.validate --checkpoint your.ckpt --steps 8 --solver euler
    python -m matcha.utils.validate --checkpoint your.ckpt --skip-utmos
    python -m matcha.utils.validate --checkpoint your.ckpt --skip-mcd

See mcd_validate_v2.py and utmos_validate.py for historical per-version measurement tables.

V23       epoch       234      334      434      609      634      714      734      739      764     1069     1139     1299     1313
speaker_000 MCD   5.19 dB  5.05 dB  4.93 dB  4.85 dB  4.87 dB  4.87 dB  4.90 dB  4.88 dB  4.81 dB  
speaker_001 MCD   3.65 dB  3.51 dB  3.45 dB  3.39 dB  3.43 dB  3.31 dB  3.39 dB  3.33 dB  3.31 dB  
speaker_002 MCD   4.00 dB  3.90 dB  3.83 dB  3.70 dB  3.75 dB  3.73 dB  3.71 dB  3.74 dB  3.66 dB  
speaker_003 MCD   3.17 dB  2.93 dB  2.87 dB  2.77 dB  2.88 dB  2.78 dB  2.85 dB  2.79 dB  2.74 dB  
speaker_004 MCD   5.12 dB  4.89 dB  4.80 dB  4.68 dB  4.76 dB  4.72 dB  4.69 dB  4.69 dB  4.67 dB  
speaker_005 MCD   4.01 dB  3.90 dB  3.86 dB  3.81 dB  3.88 dB  3.87 dB  3.84 dB  3.86 dB  3.88 dB  
speaker_006 MCD   4.33 dB  4.27 dB  4.19 dB  4.09 dB  4.16 dB  4.16 dB  4.11 dB  4.11 dB  4.09 dB  
speaker_007 MCD   5.32 dB  5.20 dB  5.18 dB  5.17 dB  5.14 dB  5.09 dB  5.14 dB  5.12 dB  5.09 dB  
speaker_008 MCD   5.26 dB  5.13 dB  5.07 dB  4.98 dB  5.03 dB  5.05 dB  5.03 dB  4.97 dB  4.93 dB  
speaker_009 MCD   4.95 dB  4.88 dB  4.82 dB  4.77 dB  4.79 dB  4.69 dB  4.79 dB  4.71 dB  4.70 dB  
speaker_010 MCD   4.32 dB  4.19 dB  4.14 dB  4.12 dB  4.15 dB  4.11 dB  4.09 dB  4.16 dB  4.18 dB  
speaker_011 MCD   5.18 dB  5.15 dB  4.96 dB  4.90 dB  4.90 dB  4.92 dB  4.93 dB  4.93 dB  4.90 dB  
speaker_012 MCD   4.31 dB  4.28 dB  4.18 dB  4.11 dB  4.15 dB  4.14 dB  4.17 dB  4.14 dB  4.12 dB  
speaker_013 MCD   6.23 dB  6.11 dB  5.98 dB  6.03 dB  5.99 dB  6.01 dB  6.03 dB  6.03 dB  6.04 dB  
speaker_014 MCD   6.05 dB  5.99 dB  5.92 dB  5.92 dB  5.94 dB  5.89 dB  5.93 dB  5.91 dB  5.87 dB  
---------------------------------------------------------------------------------------------------
Average MCD v2:   4.74 dB  4.62 dB  4.55 dB  4.49 dB  4.52 dB  4.49 dB  4.51 dB  4.49 dB  4.47 dB 

V23         epoch       234      334      434      609      634      714      734      739      764     1069     1139     1299     1313
speaker_000 UTMOS   5.19 dB  5.05 dB  4.93 dB  4.85 dB  4.87 dB  4.87 dB  4.90 dB  4.88 dB  4.81 dB  
speaker_001 UTMOS   3.65 dB  3.51 dB  3.45 dB  3.39 dB  3.43 dB  3.31 dB  3.39 dB  3.33 dB  3.31 dB  
speaker_002 UTMOS   4.00 dB  3.90 dB  3.83 dB  3.70 dB  3.75 dB  3.73 dB  3.71 dB  3.74 dB  3.66 dB  
speaker_003 UTMOS   3.17 dB  2.93 dB  2.87 dB  2.77 dB  2.88 dB  2.78 dB  2.85 dB  2.79 dB  2.74 dB  
speaker_004 UTMOS   5.12 dB  4.89 dB  4.80 dB  4.68 dB  4.76 dB  4.72 dB  4.69 dB  4.69 dB  4.67 dB  
speaker_005 UTMOS   4.01 dB  3.90 dB  3.86 dB  3.81 dB  3.88 dB  3.87 dB  3.84 dB  3.86 dB  3.88 dB  
speaker_006 UTMOS   4.33 dB  4.27 dB  4.19 dB  4.09 dB  4.16 dB  4.16 dB  4.11 dB  4.11 dB  4.09 dB  
speaker_007 UTMOS   5.32 dB  5.20 dB  5.18 dB  5.17 dB  5.14 dB  5.09 dB  5.14 dB  5.12 dB  5.09 dB  
speaker_008 UTMOS   5.26 dB  5.13 dB  5.07 dB  4.98 dB  5.03 dB  5.05 dB  5.03 dB  4.97 dB  4.93 dB  
speaker_009 UTMOS   4.95 dB  4.88 dB  4.82 dB  4.77 dB  4.79 dB  4.69 dB  4.79 dB  4.71 dB  4.70 dB  
speaker_010 UTMOS   4.32 dB  4.19 dB  4.14 dB  4.12 dB  4.15 dB  4.11 dB  4.09 dB  4.16 dB  4.18 dB  
speaker_011 UTMOS   5.18 dB  5.15 dB  4.96 dB  4.90 dB  4.90 dB  4.92 dB  4.93 dB  4.93 dB  4.90 dB  
speaker_012 UTMOS   4.31 dB  4.28 dB  4.18 dB  4.11 dB  4.15 dB  4.14 dB  4.17 dB  4.14 dB  4.12 dB  
speaker_013 UTMOS   6.23 dB  6.11 dB  5.98 dB  6.03 dB  5.99 dB  6.01 dB  6.03 dB  6.03 dB  6.04 dB  
speaker_014 UTMOS   6.05 dB  5.99 dB  5.92 dB  5.92 dB  5.94 dB  5.89 dB  5.93 dB  5.91 dB  5.87 dB  
---------------------------------------------------------------------------------------------------
Average UTMOS:      4.74 dB  4.62 dB  4.55 dB  4.49 dB  4.52 dB  4.49 dB  4.51 dB  4.49 dB  4.47 dB 
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
import torchaudio.functional as taf
from pymcd.mcd import Calculate_MCD

from matcha.inference import VOICES, DEFAULT_ODE_SOLVER, DEFAULT_NUM_STEPS, load_matcha, load_vocoder, pipeline
from matcha.utils.precompute_mels import _load_yaml_config, _resolve_path, parse_filelist

SAMPLE_OFFSET = 10
SAMPLES_PER_SPEAKER = 20
UTMOS_SAMPLE_RATE = 16000


class Calculate_MCD_24k(Calculate_MCD):
    """
    Calculate_MCD subclass that runs analysis at 24 kHz with 30 cepstral coefficients.

    The default Calculate_MCD hardcodes:
      - SAMPLING_RATE = 22050  (resamples all audio before analysis)
      - order = 13             (MCD convention, historical)
      - alpha = 0.65           (frequency warping for 22050 Hz)

    This subclass overrides all three to match the corpus sample rate and use a
    perceptually richer cepstral representation consistent with the cepstral training loss.

    It also fixes an inconsistency in the base class around C0, the 0th cepstral coefficient,
    which carries frame loudness rather than spectral shape. The base class slices C0 out when
    computing the DTW alignment path but then passes the full vectors, C0 included, to the
    distance calculation, so the reported score is sensitive to plain volume differences that
    have nothing to do with voice quality. Standard MCD sums k=1..K and excludes C0. Excluding it
    from the alignment is the right call on its own terms, since C0's dynamic range would
    otherwise dominate the cost and align the signals by loudness contour instead of spectral
    shape, so the fix is to carry that same exclusion through to the distance.
    """

    SAMPLING_RATE_24K = 24000
    ORDER = 30
    # Frequency warping coefficient for 24 kHz. The standard value for 22050 Hz is 0.65;
    # for 24 kHz the correct value is 0.66.
    ALPHA_24K = 0.66
    
    # Pymcd hardcodes the fft size to 512, which forces a floor of about 141 Hz. 
    # Most of the male speakers in my corpus have a lower pitch. 
    # The script measured male voices less accurately than female voices.
    # At 1024 the floor drops to about 70 Hz so every speaker is measured the same way.
    # This change means the values from validate.py cannot be directly compared with the values from mcd v1 or v2. 
    FFT_SIZE = 1024

    def __init__(self, MCD_mode, exclude_c0=True):
        super().__init__(MCD_mode)
        self.SAMPLING_RATE = self.SAMPLING_RATE_24K
        self.exclude_c0 = exclude_c0

    def wav2mcep_numpy(self, loaded_wav, fft_size=None):
        fft_size = self.FFT_SIZE if fft_size is None else fft_size
        _, sp, _ = pyworld.wav2world(loaded_wav.astype(np.double), fs=self.SAMPLING_RATE,
                                     frame_period=self.FRAME_PERIOD, fft_size=fft_size)
        return pysptk.sptk.mcep(sp, order=self.ORDER, alpha=self.ALPHA_24K, maxiter=0,
                                 etype=1, eps=1.0E-8, min_det=0.0, itype=3)

    def calculate_mcd_distance(self, x, y, path):
        """
        Drop C0 before measuring the distance, so the score reflects spectral shape and not loudness.

        The base class already excludes C0 when it computes `path`. Dropping it here as well makes the
        two steps consistent. `path` holds frame (row) indices, so removing a coefficient (column)
        leaves the alignment untouched.
        """
        if self.exclude_c0:
            x = x[:, 1:]
            y = y[:, 1:]
        return super().calculate_mcd_distance(x, y, path)


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

    mcd = mcd_toolbox.calculate_mcd(ref_path, gen_path)
    Path(gen_path).unlink(missing_ok=True)
    Path(ref_path).unlink(missing_ok=True)

    gt_duration = len(ref_trimmed) / ref_sr
    infer_duration = len(gen_trimmed) / sample_rate
    duration_ratio = gt_duration / infer_duration if infer_duration > 0 else 1.0
    return mcd, duration_ratio


def score_utmos(predictor, waveform: torch.Tensor, source_sr: int, device: torch.device) -> float:
    wav = waveform.detach().cpu().float()
    if wav.dim() == 2:
        wav = wav.mean(dim=0)
    elif wav.dim() > 2:
        wav = wav.squeeze()
    if source_sr != UTMOS_SAMPLE_RATE:
        wav = taf.resample(wav, source_sr, UTMOS_SAMPLE_RATE)
    wav = wav.unsqueeze(0).to(device)
    with torch.no_grad():
        score = predictor(wav, UTMOS_SAMPLE_RATE)
    return float(score.squeeze().item())


def main():
    parser = argparse.ArgumentParser(description="Combined MCD + UTMOS validation, sharing a single inference pass per sample")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data-config", default="configs/data/corpus-24k.yaml")
    parser.add_argument("--vocoder", default="vocos", choices=["vocos"])
    parser.add_argument("--solver", type=str, default=DEFAULT_ODE_SOLVER)
    parser.add_argument("--steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--skip-mcd", action="store_true", help="Skip MCD scoring")
    parser.add_argument("--skip-utmos", action="store_true", help="Skip UTMOS scoring")
    parser.add_argument("--c0", choices=["exclude", "include"], default="exclude",
                        help="Whether the MCD distance counts C0, the loudness coefficient. "
                             "Standard MCD excludes it, so the score reflects spectral shape rather than volume. "
                             "Use 'include' to reproduce pymcd's own behaviour.")
    args = parser.parse_args()

    if args.skip_mcd and args.skip_utmos:
        parser.error("--skip-mcd and --skip-utmos cannot both be set")

    ckpt_name = Path(args.checkpoint).stem
    print(f"Processing {ckpt_name}...")

    cfg = _load_yaml_config(Path(args.data_config).resolve())
    valid_filelist = _resolve_path(str(cfg["valid_filelist_path"]))
    sample_rate = int(cfg["sample_rate"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_matcha("custom_model", args.checkpoint)
    model.decoder.solver = args.solver
    vocoder = load_vocoder(args.vocoder)

    mcd_toolbox = Calculate_MCD_24k(MCD_mode="dtw", exclude_c0=args.c0 == "exclude") if not args.skip_mcd else None

    predictor = None
    if not args.skip_utmos:
        predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
        predictor = predictor.to(device).eval()

    for voice in VOICES:
        voice["scale_correction"] = 1.0

    speaker_mcd_scores: dict[str, tuple[float, float]] = {}  # voice_id -> (avg_mcd, avg_duration_ratio)
    speaker_utmos_scores: dict[str, float] = {}

    for voice in VOICES:
        spk_id = int(voice["id"])
        language = voice["lang"]
        samples = pick_samples(valid_filelist, voice["id"])
        if not samples:
            print(f"[!] No samples found for speaker {spk_id}, skipping.")
            continue

        mcd_scores = []
        duration_ratios = []
        utmos_scores = []
        for text, gt_wav_path in samples:
            waveform = pipeline(model, vocoder, text, spk_id, None, args.steps)
            if mcd_toolbox is not None:
                mcd, duration_ratio = compute_mcd(waveform, gt_wav_path, mcd_toolbox, sample_rate)
                mcd_scores.append(mcd)
                duration_ratios.append(duration_ratio)
            if predictor is not None:
                utmos_scores.append(score_utmos(predictor, waveform, sample_rate, device))

        if mcd_scores:
            speaker_mcd_scores[voice["id"]] = (sum(mcd_scores) / len(mcd_scores), sum(duration_ratios) / len(duration_ratios))
        if utmos_scores:
            speaker_utmos_scores[voice["id"]] = sum(utmos_scores) / len(utmos_scores)

    reported_ids = [v["id"] for v in VOICES if v["id"] in speaker_mcd_scores or v["id"] in speaker_utmos_scores]

    print()
    for voice_id in reported_ids:
        label = f"speaker_{int(voice_id):03d}"
        columns = [f"{label:<40}"]
        if voice_id in speaker_mcd_scores:
            columns.append(f"MCD: {speaker_mcd_scores[voice_id][0]:5.2f} dB")
        if voice_id in speaker_utmos_scores:
            columns.append(f"UTMOS: {speaker_utmos_scores[voice_id]:5.2f}")
        if voice_id in speaker_mcd_scores:
            columns.append(f"duration ratio: {speaker_mcd_scores[voice_id][1]:.2f}")
        print("   ".join(columns))

    print("-" * 85)
    columns = [f"{'Average':<40}"]
    if speaker_mcd_scores:
        all_mcds = [mcd for mcd, _ in speaker_mcd_scores.values()]
        columns.append(f"MCD: {sum(all_mcds) / len(all_mcds):5.2f} dB")
    if speaker_utmos_scores:
        columns.append(f"UTMOS: {sum(speaker_utmos_scores.values()) / len(speaker_utmos_scores):5.2f}")
    print("   ".join(columns))

    print()
    print(f"Completed {ckpt_name}")


if __name__ == "__main__":
    main()
