"""
Validation script: MCD (24 kHz, 30 coefficients).

MCD (reference-based, lower is better):
  Similar to mcd_validate_v2.py except it excludes C0 (loudness) from the distance calculation and the FFT is 
  1024 instead of 512 which lowers the pitch floor to 70Hz instead of 144Hz.
  Uses a subclass of Calculate_MCD that overrides:
    - SAMPLING_RATE: 24000 Hz  (matches the corpus and the model, avoids resampling to 22050)
    - order:         30        (captures the full perceptually relevant cepstral range, vs the default 13)
    - alpha:         0.66      (frequency warping coefficient tuned for 24 kHz, vs the default 0.65 for 22050 Hz)
  MCD values from this script are NOT directly comparable to mcd_validate.py (the 13-coefficient,
  22050 Hz variant) values, because a different order and sample rate produce a different cepstral scale.

Picks SAMPLES_PER_SPEAKER samples per speaker, starting at SAMPLE_OFFSET, from validate.csv,
forcing scale_correction to 1.0 for every voice before running.

Usage:
    python -m matcha.utils.mcd_validate_v3 --checkpoint your.ckpt
    python -m matcha.utils.mcd_validate_v3 --checkpoint your.ckpt --steps 4 --solver midpoint

See mcd_validate_v2.py for historical per-version measurement tables.

All the figures below were taken with heun3 / 5 steps. 
This version of the script shows better results with heun3 / 5 than with midpoint / 4, 
which agrees with my subjective tests. The results from V2 were always better with midpoint / 4,
which is why I did not trust those results too much.

V23       epoch      5229  
speaker_000 MCD   4.35 dB  
speaker_001 MCD   3.58 dB  
speaker_002 MCD   4.11 dB  
speaker_003 MCD   3.29 dB  
speaker_004 MCD   4.31 dB  
speaker_005 MCD   3.29 dB  
speaker_006 MCD   4.22 dB  
speaker_007 MCD   4.31 dB  
speaker_008 MCD   5.35 dB  
speaker_009 MCD   4.52 dB  
speaker_010 MCD   3.33 dB  
speaker_011 MCD   4.79 dB  
speaker_012 MCD   3.64 dB  
speaker_013 MCD   6.04 dB  
speaker_014 MCD   5.16 dB  
-------------------------
Average MCD:      4.29 dB 

I do not know for sure why the v24 figures are so much lower than in v23. 
And am not sure I can hear the difference in subjective tests. 
I think the voices are brighter and maybe I hear less artifacts, but it may be just confirmation bias.
The training losses are for sure much lower than in v23. Claude Opus could not explain the reason for the lower MCD.
The most likely explanation in my opinion is the TIMESTEP_SAMPLING_SCALE and TIMESTEP_SAMPLING_LOCATION values
I used in v24. Together with the SWAY value, they push the model to sample more timesteps in the late part of the 
trajectory, improving the accuracy there. It counts more than the accuracy in the middle or early in the trajectory,
because there are no more steps that could correct an inaccurate velocity at that point.
I have captured all changes made in v24 in the config file, v24.yaml, and we reviewed 
them but we still don't have an explanation.
I already measured how much EMA helps. It is around 0.02 dB, so EMA is not the explanation for the 0.4 dB reduction.

V24       epoch       124      279      479      749    1059c    1089b    1134c    1474b    1634b    1739b    1834b    2004b
speaker_000 MCD   4.64 dB  4.36 dB  4.25 dB  4.19 dB  4.10 dB  4.08 dB  4.08 dB  4.05 dB  3.98 dB  4.00 dB  3.98 dB  4.00 dB
speaker_001 MCD   3.63 dB  3.44 dB  3.38 dB  3.30 dB  3.23 dB  3.27 dB  3.25 dB  3.25 dB  3.23 dB  3.21 dB  3.20 dB  3.16 dB
speaker_002 MCD   4.08 dB  3.88 dB  3.75 dB  3.71 dB  3.66 dB  3.64 dB  3.64 dB  3.63 dB  3.61 dB  3.61 dB  3.58 dB  3.53 dB
speaker_003 MCD   3.06 dB  2.81 dB  2.75 dB  2.73 dB  2.70 dB  2.70 dB  2.71 dB  2.68 dB  2.71 dB  2.68 dB  2.64 dB  2.63 dB
speaker_004 MCD   4.93 dB  4.61 dB  4.34 dB  4.24 dB  4.22 dB  4.21 dB  4.24 dB  4.18 dB  4.16 dB  4.19 dB  4.07 dB  4.02 dB
speaker_005 MCD   3.52 dB  3.31 dB  3.22 dB  3.15 dB  3.14 dB  3.18 dB  3.15 dB  3.14 dB  3.10 dB  3.12 dB  3.04 dB  3.11 dB
speaker_006 MCD   4.01 dB  3.95 dB  3.79 dB  3.82 dB  3.78 dB  3.77 dB  3.75 dB  3.75 dB  3.76 dB  3.76 dB  3.73 dB  3.71 dB
speaker_007 MCD   4.14 dB  3.92 dB  3.82 dB  3.77 dB  3.74 dB  3.76 dB  3.71 dB  3.72 dB  3.73 dB  3.70 dB  3.70 dB  3.71 dB
speaker_008 MCD   5.19 dB  5.05 dB  4.94 dB  4.86 dB  4.85 dB  4.87 dB  4.82 dB  4.85 dB  4.84 dB  4.81 dB  4.79 dB  4.81 dB
speaker_009 MCD   4.32 dB  4.17 dB  4.07 dB  4.03 dB  4.01 dB  4.00 dB  4.03 dB  3.99 dB  3.99 dB  3.99 dB  3.98 dB  3.96 dB
speaker_010 MCD   3.58 dB  3.34 dB  3.19 dB  3.18 dB  3.13 dB  3.12 dB  3.08 dB  3.07 dB  3.09 dB  3.07 dB  3.06 dB  3.04 dB
speaker_011 MCD   4.64 dB  4.61 dB  4.34 dB  4.28 dB  4.24 dB  4.31 dB  4.27 dB  4.26 dB  4.30 dB  4.28 dB  4.21 dB  4.22 dB
speaker_012 MCD   3.60 dB  3.31 dB  3.17 dB  3.11 dB  3.05 dB  3.03 dB  3.03 dB  3.04 dB  2.99 dB  3.01 dB  2.95 dB  2.97 dB
speaker_013 MCD   6.01 dB  5.76 dB  5.70 dB  5.61 dB  5.63 dB  5.58 dB  5.62 dB  5.61 dB  5.61 dB  5.54 dB  5.56 dB  5.55 dB
speaker_014 MCD   5.09 dB  4.92 dB  4.85 dB  4.83 dB  4.79 dB  4.82 dB  4.83 dB  4.82 dB  4.78 dB  4.80 dB  4.76 dB  4.76 dB
----------------------------------------------------------------------------------------------------------------------------
Average MCD v3:   4.30 dB  4.10 dB  3.97 dB  3.92 dB  3.88 dB  3.89 dB  3.88 dB  3.87 dB  3.86 dB  3.85 dB  3.82 dB  3.81 dB

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


def main():
    parser = argparse.ArgumentParser(description="MCD validation")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data-config", default="configs/data/corpus-24k.yaml")
    parser.add_argument("--vocoder", default="vocos", choices=["vocos"])
    parser.add_argument("--solver", type=str, default=DEFAULT_ODE_SOLVER)
    parser.add_argument("--steps", type=int, default=DEFAULT_NUM_STEPS)
    parser.add_argument("--c0", choices=["exclude", "include"], default="exclude",
                        help="Whether the MCD distance counts C0, the loudness coefficient. "
                             "Standard MCD excludes it, so the score reflects spectral shape rather than volume. "
                             "Use 'include' to reproduce pymcd's own behaviour.")
    parser.add_argument("--no-ema", action="store_true",
                        help="Use the live Decoder weights even if the checkpoint holds their moving average.")
    args = parser.parse_args()

    ckpt_name = Path(args.checkpoint).stem
    print(f"Processing {ckpt_name}...")

    cfg = _load_yaml_config(Path(args.data_config).resolve())
    valid_filelist = _resolve_path(str(cfg["valid_filelist_path"]))
    sample_rate = int(cfg["sample_rate"])

    model = load_matcha("custom_model", args.checkpoint, use_decoder_ema=not args.no_ema)
    model.decoder.solver = args.solver
    vocoder = load_vocoder(args.vocoder)

    mcd_toolbox = Calculate_MCD_24k(MCD_mode="dtw", exclude_c0=args.c0 == "exclude")

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
        print(f"{label:<40}   MCD: {avg_mcd:5.2f} dB   duration ratio: {avg_ratio:.2f}")

    print("-" * 85)
    if speaker_mcd_scores:
        all_mcds = [mcd for mcd, _ in speaker_mcd_scores.values()]
        print(f"{'Average':<40}   MCD: {sum(all_mcds) / len(all_mcds):5.2f} dB")

    print()
    print(f"Completed {ckpt_name}")


if __name__ == "__main__":
    main()
