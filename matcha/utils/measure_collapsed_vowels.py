"""
Measure, per speaker, how often the alignment squeezes a vowel down to the minimum length.

Every voiced phoneme is written as three symbols (opening separator, phoneme, closing separator), and the
alignment gives each symbol at least one fine frame of 5.3ms. A vowel whose three symbols get 3 frames in
total lasts 16ms, which no speaker produces. Such vowels mark places where the alignment ran through the
text without finding the sound, usually because a neighbouring symbol took over its audio.

The alignment is computed the same way training computes it: from the Encoder output of the checkpoint and
the precomputed fine mel of each recording. It runs on the CPU, so it can be used while training holds the GPU.

Usage:
  python -m matcha.utils.measure_collapsed_vowels -i configs/data/corpus-24k.yaml -c <checkpoint.ckpt>
  python -m matcha.utils.measure_collapsed_vowels -i configs/data/corpus-24k.yaml -c <checkpoint.ckpt> \
      --split train --files_per_speaker 300
"""

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from matcha.utils.alignment_diagnostics import (
    LoadedEncoder,
    Utterance,
    ground_truth_alignment,
    load_encoder,
    load_fine_mel,
    read_filelist,
    vowel_core_positions,
)

# A vowel group at or below this many fine frames is shorter than any real vowel (3 frames = 16ms).
COLLAPSED_VOWEL_MAX_FRAMES = 3


def count_collapsed_vowels(utterance: Utterance, durations: np.ndarray) -> Tuple[int, int, int]:
    """Returns the number of vowels, how many of them collapsed, and the longest run of consecutive collapsed vowels."""
    vowel_count = 0
    collapsed_count = 0
    current_run = 0
    longest_run = 0
    for position in vowel_core_positions(utterance.phoneme_ids):
        vowel_frames = durations[position - 1] + durations[position] + durations[position + 1]
        vowel_count += 1
        if vowel_frames <= COLLAPSED_VOWEL_MAX_FRAMES:
            collapsed_count += 1
            current_run += 1
            longest_run = max(longest_run, current_run)
        else:
            current_run = 0
    return vowel_count, collapsed_count, longest_run


def measure_file(loaded: LoadedEncoder, utterance: Utterance, mel_dir: Path) -> Tuple[int, int, int]:
    mel = load_fine_mel(mel_dir, utterance.relative_path)
    durations = ground_truth_alignment(loaded, utterance, mel)
    return count_collapsed_vowels(utterance, durations)


def main():
    parser = argparse.ArgumentParser(description="Measure how often the alignment collapses vowels, per speaker.")
    parser.add_argument("-i", "--data-config", required=True, help="Path to data YAML, e.g. configs/data/corpus-24k.yaml")
    parser.add_argument("-c", "--checkpoint", required=True, help="Checkpoint whose Encoder computes the alignment")
    parser.add_argument("--split", choices=["validate", "train"], default="validate")
    parser.add_argument("--files_per_speaker", type=int, default=100)
    parser.add_argument("--threads", type=int, default=8, help="CPU threads for the Encoder forward pass")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    data_config = OmegaConf.load(args.data_config)
    if args.split == "validate":
        filelist_path = Path(data_config.valid_filelist_path)
    else:
        filelist_path = Path(data_config.train_filelist_path)
    mel_dir = Path(data_config.mel_dir)

    loaded = load_encoder(args.checkpoint)
    utterances_by_speaker = read_filelist(filelist_path)

    print(f"Checkpoint: {args.checkpoint} (epoch {loaded.epoch})")
    print(f"A vowel is collapsed when its three symbols get {COLLAPSED_VOWEL_MAX_FRAMES} fine frames or fewer.")
    print(f"{'speaker':>7} {'files':>5} {'vowels':>7} {'collapsed':>9} {'files with 2+ in a row':>22}  worst file")
    for speaker in sorted(utterances_by_speaker):
        utterances = utterances_by_speaker[speaker][:args.files_per_speaker]
        total_vowels = 0
        total_collapsed = 0
        files_with_consecutive_collapses = 0
        worst_run = 0
        worst_file = "-"
        for utterance in utterances:
            vowel_count, collapsed_count, longest_run = measure_file(loaded, utterance, mel_dir)
            total_vowels += vowel_count
            total_collapsed += collapsed_count
            if longest_run >= 2:
                files_with_consecutive_collapses += 1
            if longest_run > worst_run:
                worst_run = longest_run
                worst_file = f"{utterance.relative_path} ({longest_run} in a row)"
        collapsed_share = 100 * total_collapsed / max(1, total_vowels)
        print(
            f"{speaker:7d} {len(utterances):5d} {total_vowels:7d} {collapsed_share:8.1f}% "
            f"{files_with_consecutive_collapses:22d}  {worst_file}",
            flush=True,
        )


if __name__ == "__main__":
    main()
