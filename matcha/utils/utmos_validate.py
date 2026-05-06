"""
UTokyo-SaruLab mean opinion score (UTMOS) validation script.
https://arxiv.org/abs/2204.02152
https://github.com/sarulab-speech/UTMOS22

Loads the model once, picks 20 samples per speaker (skipping the first 10) from validate.csv,
runs inference for each speaker using their own ground truth lines, and reports average
predicted naturalness score per speaker.

UTMOS is a reference-free predicted MOS on a 1-5 scale. Higher is better. Production-quality
TTS usually lands above 4.0. The model is downloaded on first run via torch.hub from
tarepan/SpeechMOS and cached under ~/.cache/torch/hub.

Usage:
    python -m matcha.utils.utmos_validate --checkpoint logs/train/v18/averaged.ckpt

Edit the constants at the top to change anything other than the checkpoint.

V18
----------------------------------------------------------------------------------------------
                     044   064   094   164   199   224   264
speaker_000 UTMOS:  3.41  3.43  3.72  3.91  4.02  4.04  4.04
speaker_001 UTMOS:  3.86  3.94  4.16  4.22  4.31  4.25  4.31
speaker_002 UTMOS:  3.42  3.48  3.84  3.88  3.77  3.86  3.94
speaker_003 UTMOS:  3.32  3.38  3.64  3.86  3.95  3.96  3.88
speaker_004 UTMOS:  2.77  2.79  3.11  3.34  3.45  3.52  3.57
speaker_005 UTMOS:  3.40  3.35  3.57  3.81  3.85  3.80  3.94
speaker_006 UTMOS:  2.74  2.84  2.85  3.03  3.11  3.08  3.10
speaker_007 UTMOS:  2.59  2.68  2.94  3.17  3.20  3.21  3.27
speaker_008 UTMOS:  2.26  2.38  2.58  2.95  2.97  3.04  3.11
speaker_009 UTMOS:  2.79  2.87  3.03  3.25  3.22  3.29  3.28
--------------------------------------------------------------------------------------------------------------
Average UTMOS:      3.06  3.11  3.34  3.54  3.59  3.61  3.65

"""

import argparse
from pathlib import Path

import torch
import torchaudio.functional as taf

from matcha.inference import VOICES, load_matcha, load_vocoder, pipeline
from matcha.utils.precompute_mels import _load_yaml_config, _resolve_path, parse_filelist

DATA_CONFIG = "configs/data/corpus-24k.yaml"
VOCODER = "vocos"
STEPS = 20
SAMPLE_OFFSET = 10
SAMPLES_PER_SPEAKER = 20
UTMOS_SAMPLE_RATE = 16000


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
    parser = argparse.ArgumentParser(description="UTMOS validation: predicted naturalness score per speaker")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    ckpt_name = Path(args.checkpoint).stem
    print(f"Processing {ckpt_name}...")

    cfg = _load_yaml_config(Path(DATA_CONFIG).resolve())
    valid_filelist = _resolve_path(str(cfg["valid_filelist_path"]))
    sample_rate = int(cfg["sample_rate"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_matcha("custom_model", args.checkpoint)
    vocoder = load_vocoder(VOCODER)

    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    predictor = predictor.to(device).eval()

    for voice in VOICES:
        voice["scale_correction"] = 1.0

    speaker_scores: dict[str, float] = {}

    for voice in VOICES:
        spk_id = int(voice["id"])
        language = voice["lang"]
        samples = pick_samples(valid_filelist, voice["id"])
        if not samples:
            print(f"[!] No samples found for speaker {spk_id}, skipping.")
            continue

        scores = []
        for text, _gt_wav_path in samples:
            waveform = pipeline(model, vocoder, text, language, spk_id, None, STEPS)
            scores.append(score_utmos(predictor, waveform, sample_rate, device))
        speaker_scores[voice["id"]] = sum(scores) / len(scores)

    print()
    for voice_id, avg_score in speaker_scores.items():
        label = f"speaker_{int(voice_id):03d}"
        print(f"{label:<40} UTMOS: {avg_score:5.2f}")
    print("-" * 70)
    if speaker_scores:
        print(f"{'Average':<40} UTMOS: {sum(speaker_scores.values()) / len(speaker_scores):5.2f}")
    print()
    print(f"Completed {ckpt_name}")


if __name__ == "__main__":
    main()
