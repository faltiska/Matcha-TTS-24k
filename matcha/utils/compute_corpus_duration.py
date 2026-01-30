"""
Compute total audio duration for train and validation sets.

Usage:
  python -m matcha.utils.compute_corpus_duration data/corpus-small-24k/train.csv data/corpus-small-24k/validate.csv
"""

import argparse
import wave
from pathlib import Path


def parse_filelist(filelist_path: Path, split_char: str = "|"):
    with open(filelist_path, encoding="utf-8") as f:
        return [line.strip().split(split_char) for line in f if line.strip()]


def compute_duration(filelist_path: Path, label: str):
    entries = parse_filelist(filelist_path)
    filelist_dir = filelist_path.parent
    total_duration = 0.0
    total = len(entries)
    
    for i, parts in enumerate(entries, start=1):
        if not parts:
            continue
        wav_path = filelist_dir / "wav" / f"{parts[0]}.wav"
        try:
            with wave.open(str(wav_path), 'rb') as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                total_duration += frames / float(rate)
        except Exception:
            pass
        print(f"\r[{label}] {i}/{total} processed...", end="", flush=True)
    
    print()
    return total_duration


def main():
    parser = argparse.ArgumentParser(description="Compute total audio duration for corpus.")
    parser.add_argument("train_csv", help="Path to train CSV file")
    parser.add_argument("validate_csv", help="Path to validate CSV file")
    args = parser.parse_args()

    train_path = Path(args.train_csv).resolve()
    valid_path = Path(args.validate_csv).resolve()

    train_duration = compute_duration(train_path, "Train")
    valid_duration = compute_duration(valid_path, "Valid")

    print(f"Train set: {int(train_duration // 3600)}h {int((train_duration % 3600) // 60)}m {int(train_duration % 60)}s ({train_duration:.2f}s)")
    print(f"Valid set: {int(valid_duration // 3600)}h {int((valid_duration % 3600) // 60)}m {int(valid_duration % 60)}s ({valid_duration:.2f}s)")


if __name__ == "__main__":
    main()
