"""
MCD validation script.

Loads the model once, picks 20 samples per speaker (skipping the first 10) from validate.csv,
runs inference for each speaker using their own ground truth lines, and reports average MCD per speaker.

Usage:
    python -m matcha.utils.mcd_validate --checkpoint averaged.ckpt
    python -m matcha.utils.mcd_validate --checkpoint averaged.ckpt --data-config configs/data/corpus-small-24k.yaml --vocoder vocos --steps 20
    
logs/train/v4/averaged1.ckpt
logs/train/v4/averaged2.ckpt
logs/train/v4/checkpoint_epoch=1189.ckpt    

                  v4/avg1      v4/avg2   
speaker_000 MCD:  4.66 dB      4.60 dB
speaker_001 MCD:  3.26 dB      3.25 dB
speaker_002 MCD:  3.51 dB      3.55 dB
speaker_003 MCD:  2.65 dB      2.65 dB
speaker_004 MCD:  4.83 dB      4.85 dB
speaker_005 MCD:  3.73 dB      3.70 dB
speaker_006 MCD:  3.76 dB      3.71 dB
speaker_007 MCD:  4.90 dB      4.88 dB
speaker_008 MCD:  4.73 dB      4.72 dB
speaker_009 MCD:  4.33 dB      4.30 dB

"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="pyworld")

import argparse
import tempfile
from pathlib import Path

import soundfile as sf
import torch
import torchaudio as ta
from pymcd.mcd import Calculate_MCD

from matcha.inference import VOICES, load_matcha, load_vocoder, pipeline
from matcha.utils.precompute_corpus import _load_yaml_config, _resolve_path, parse_filelist

SAMPLE_OFFSET = 10
SAMPLES_PER_SPEAKER = 20

def pick_samples(valid_filelist: Path, speaker_id: str) -> list[tuple[str, Path]]:
    """Return SAMPLES_PER_SPEAKER rows starting at SAMPLE_OFFSET from validate.csv for the given speaker."""
    all_rows = parse_filelist(valid_filelist)
    samples = []
    skipped = 0
    for rel_path, spk_id, _lang, text in all_rows:
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
    parser = argparse.ArgumentParser(description="MCD validation: compare TTS output to ground truth per speaker")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--data-config", default="configs/data/corpus-small-24k.yaml")
    parser.add_argument("--vocoder", default="vocos", choices=["vocos", "bigvgan"])
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()

    ckpt_name = Path(args.checkpoint).stem
    print(f"Processing {ckpt_name}...")

    cfg = _load_yaml_config(Path(args.data_config).resolve())
    valid_filelist = _resolve_path(str(cfg["valid_filelist_path"]))
    sample_rate = int(cfg["sample_rate"])

    model = load_matcha("custom_model", args.checkpoint)
    vocoder = load_vocoder(args.vocoder)
    mcd_toolbox = Calculate_MCD(MCD_mode="dtw")

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
            waveform = pipeline(model, vocoder, text, language, spk_id, None, args.steps, voice["default_scale"])
            mcd, duration_ratio = compute_mcd(waveform, gt_wav_path, mcd_toolbox, sample_rate)
            mcd_scores.append(mcd)
            duration_ratios.append(duration_ratio)
        speaker_mcd_scores[voice["id"]] = (sum(mcd_scores) / len(mcd_scores), sum(duration_ratios) / len(duration_ratios))

    print()
    for voice_id, (avg_mcd, avg_ratio) in speaker_mcd_scores.items():
        voice = next(v for v in VOICES if v["id"] == voice_id)
        suggested_scale = voice["default_scale"] * avg_ratio
        label = f"speaker_{int(voice_id):03d}"
        print(f"{label:<40} MCD: {avg_mcd:5.2f} dB   duration ratio: {avg_ratio:.2f}   suggested scale: {suggested_scale:.2f}")
    print("-" * 70)
    all_mcds = [mcd for mcd, _ in speaker_mcd_scores.values()]
    if all_mcds:
        print(f"{'Average':<40} MCD: {sum(all_mcds) / len(all_mcds):5.2f} dB")
    print()
    print(f"Completed {ckpt_name}")


if __name__ == "__main__":
    main()
