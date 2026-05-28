"""
UTMOS benchmark for short utterances.

Synthesizes a curated set of short utterances per language (en-us, en-gb, fr-fr, it, ro)
using each trained speaker, saves the audio to a subfolder for subjective listening, and
reports predicted MOS per speaker, per length bucket.

UTMOS is reference-free, so no ground-truth wavs are needed — only the synthesized output
is scored.

The point of this script is to expose how the model handles very short inputs ("I.", "Me.")
versus typical sentence lengths. With a healthy model, UTMOS should be comparable across
buckets. With a model that fails on short inputs, the "short" column will be visibly lower
than the "long" column.

Usage:
    python -m matcha.utils.utmos_short_utterances --checkpoint logs/train/v20/some.ckpt

The synthesized wav files are kept under:
    logs/utmos_short/<checkpoint_stem>/
so you can listen to them yourself for a subjective check alongside the UTMOS numbers.
"""

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio.functional as taf

from matcha.inference import VOICES, SAMPLE_RATE, load_matcha, load_vocoder, pipeline

VOCODER = "vocos"
UTMOS_SAMPLE_RATE = 16000
OUTPUT_DIR = Path("logs/utmos_short")

# Test set: each language → bucket name → list of test utterances.
# Two buckets: "short" (single-word and brief exclamations) and "long" (medium-length
# sentences). The "long" bucket gives a baseline for what good quality looks like on a
# normal-length sentence, against which the "short" bucket scores can be compared.
TEST_UTTERANCES = {
    "en-us": {
        "short": [
            "I.",
            "Me.",
            "Yes.",
            "No.",
            "Hi.",
            "Help.",
            "Know!",
            "Quit!",
            "Out!",
            "Come!",
            "Oh!",
            "Eh!",
        ],
        "long": [
            "Can you help me with this?",
            "Where do you want to go today?",
            "I have something to tell you.",
            "It was raining when she left.",
        ],
    },
    "en-gb": {
        "short": [
            "I.",
            "Me.",
            "Yes.",
            "No.",
            "Hi.",
            "Help.",
            "Know!",
            "Quit!",
            "Out!",
            "Come!",
            "Oh!",
            "Eh!",
        ],
        "long": [
            "Can you help me with this?",
            "Where do you want to go today?",
            "I have something to tell you.",
            "It was raining when she left.",
        ],
    },
    "fr-fr": {
        "short": [
            "Oui.",
            "Non.",
            "Stop.",
            "Viens.",
            "Pars!",
            "Va!",
            "Eh!",
            "Oh!",
            "Ah!",
            "Quoi?",
            "Hein?",
            "Tiens!",
        ],
        "long": [
            "Peux-tu m'aider avec cela ?",
            "Où veux-tu aller aujourd'hui ?",
            "J'ai quelque chose à te dire.",
            "Il pleuvait quand elle est partie.",
        ],
    },
    "it": {
        "short": [
            "Sì.",
            "No.",
            "Vai.",
            "Vieni.",
            "Stop!",
            "Via!",
            "Eh!",
            "Ah!",
            "Oh!",
            "Ehi!",
            "Cosa?",
            "Basta!",
        ],
        "long": [
            "Puoi aiutarmi con questo?",
            "Dove vuoi andare oggi?",
            "Ho qualcosa da dirti.",
            "Pioveva quando lei è partita.",
        ],
    },
    "ro": {
        "short": [
            "Da.",
            "Nu.",
            "Stai.",
            "Vino.",
            "Hai!",
            "Of!",
            "Au!",
            "Eu!",
            "Ioi!",
            "Taci!",
            "Fiu.",
            "Gol.",
            "Dud.",
        ],
        "long": [
            "Poți să mă ajuți cu asta?",
            "Unde vrei să mergi azi?",
            "Am ceva să-ți spun.",
            "Ploua când a plecat ea.",
        ],
    },
}

BUCKET_ORDER = ["short", "long"]


def _sanitize_for_filename(text: str, max_length: int = 30) -> str:
    """Return a filename-safe fragment derived from the test utterance."""
    sanitized = re.sub(r'[^A-Za-z0-9]+', '_', text).strip('_')
    return sanitized[:max_length]


def _waveform_to_numpy(waveform) -> np.ndarray:
    """Normalize a synthesized waveform to a 1D mono numpy float32 array suitable for sf.write."""
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().float().numpy()
    waveform = np.asarray(waveform)
    if waveform.ndim == 2:
        if waveform.shape[0] == 1:
            waveform = waveform.squeeze(0)
        else:
            waveform = waveform.mean(axis=0)
    return waveform.astype(np.float32)


def score_utmos(predictor, waveform, source_sr: int, device: torch.device) -> float:
    if isinstance(waveform, np.ndarray):
        wav = torch.from_numpy(waveform).float()
    else:
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
    parser = argparse.ArgumentParser(description="UTMOS benchmark for short utterances")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt_name = ckpt_path.stem
    print(f"Processing {ckpt_name}...")

    output_root = OUTPUT_DIR / ckpt_name
    output_root.mkdir(parents=True, exist_ok=True)
    print(f"Saving wav files to: {output_root.resolve()}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_matcha("custom_model", str(ckpt_path))
    vocoder = load_vocoder(VOCODER)

    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True)
    predictor = predictor.to(device).eval()

    # Disable per-speaker length-scale corrections so the benchmark exercises the raw model.
    for voice in VOICES:
        voice["scale_correction"] = 1.0

    speaker_bucket_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
    per_utterance_records = []

    for voice in VOICES:
        spk_id = int(voice["id"])
        language = voice["lang"]

        if language not in TEST_UTTERANCES:
            print(f"[!] No test utterances for language {language!r} (speaker {spk_id} {voice['name']}). Skipping.")
            continue

        speaker_label = f"speaker_{spk_id:02d}_{voice['name']}_{language}"
        print(f"\n{speaker_label}")

        utterances_by_bucket = TEST_UTTERANCES[language]
        for bucket_name in BUCKET_ORDER:
            for utterance_index, text in enumerate(utterances_by_bucket[bucket_name]):
                try:
                    waveform = pipeline(model, vocoder, text, spk_id)
                except Exception as exc:
                    print(f"  [error] {bucket_name}[{utterance_index}] {text!r}: {exc}")
                    continue

                score = score_utmos(predictor, waveform, SAMPLE_RATE, device)

                wav_filename = (
                    f"spk{spk_id:02d}_{language}_"
                    f"{bucket_name}_{utterance_index:02d}_"
                    f"{_sanitize_for_filename(text)}.wav"
                )
                waveform_np = _waveform_to_numpy(waveform)
                sf.write(
                    str(output_root / wav_filename),
                    waveform_np,
                    SAMPLE_RATE,
                    format="WAV",
                    subtype="PCM_16",
                )

                speaker_bucket_scores[(voice["id"], bucket_name)].append(score)
                per_utterance_records.append({
                    "speaker_id": voice["id"],
                    "name": voice["name"],
                    "lang": language,
                    "bucket": bucket_name,
                    "text": text,
                    "score": score,
                })
                print(f"  {bucket_name:<14} [{utterance_index}] {score:5.2f}  {text!r}")

    # Summary table
    print()
    print("=" * 110)
    print("Summary: mean UTMOS per speaker per length bucket")
    print("=" * 110)
    header = f"{'Speaker':<28} {'Lang':<8}"
    for bucket in BUCKET_ORDER:
        header += f"{bucket:<16}"
    print(header)
    print("-" * 110)

    for voice in VOICES:
        if voice["lang"] not in TEST_UTTERANCES:
            continue
        speaker_label = f"speaker_{int(voice['id']):02d}_{voice['name']}"
        row = f"{speaker_label:<28} {voice['lang']:<8}"
        for bucket in BUCKET_ORDER:
            scores = speaker_bucket_scores.get((voice["id"], bucket), [])
            if scores:
                row += f"{sum(scores) / len(scores):<16.2f}"
            else:
                row += f"{'-':<16}"
        print(row)

    # Average per bucket across all speakers (gives the headline diagnostic line)
    print("-" * 110)
    overall_row = f"{'(average across speakers)':<28} {'':<8}"
    for bucket in BUCKET_ORDER:
        all_scores_for_bucket = [
            score for (_, bk), scores in speaker_bucket_scores.items() if bk == bucket for score in scores
        ]
        if all_scores_for_bucket:
            overall_row += f"{sum(all_scores_for_bucket) / len(all_scores_for_bucket):<16.2f}"
        else:
            overall_row += f"{'-':<16}"
    print(overall_row)
    print("=" * 110)

    print(f"\nDone. {len(per_utterance_records)} utterances synthesized.")
    print(f"Listen yourself by playing files under: {output_root.resolve()}")

    # Worst 5 by UTMOS score — useful for finding the most problematic outputs to listen to first
    worst = sorted(per_utterance_records, key=lambda r: r["score"])[:5]
    print("\nWorst 5 utterances by UTMOS score:")
    for record in worst:
        print(
            f"  {record['score']:5.2f}  spk{int(record['speaker_id']):02d} "
            f"{record['name']:<10} ({record['lang']:<6}) "
            f"{record['bucket']:<14} {record['text']!r}"
        )


if __name__ == "__main__":
    main()
