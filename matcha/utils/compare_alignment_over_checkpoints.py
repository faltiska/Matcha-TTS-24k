"""
Follow the alignment across checkpoints on the same validation files.

Per checkpoint:
  - collapsed vowel share (see measure_collapsed_vowels.py),
  - duration error: mean distance between the Duration Predictor output and the alignment, on the log scale
    the duration loss uses,
  - for every tracked recording, where each word and word gap lands on the ground truth.
Per pair of consecutive checkpoints:
  - how many symbol boundaries moved by more than 2 fine frames, scaled to 100 epochs, so intervals of different
    lengths compare. A high value means the alignment is still fluid; near zero means it has settled.

Runs on the CPU, so it can be used while training holds the GPU.

Usage:
  python -m matcha.utils.compare_alignment_over_checkpoints -i configs/data/corpus-24k.yaml \
      -c <ckpt_a> <ckpt_b> [...] --track 4/1423 4/1428 [--files_per_speaker 50]
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from omegaconf import OmegaConf

from matcha.utils.alignment_diagnostics import (
    FINE_FRAME_SECONDS,
    alignment_scores,
    find_utterance,
    load_encoder,
    load_fine_mel,
    monotonic_alignment_durations,
    read_filelist,
    run_encoder,
    word_segments,
)
from matcha.utils.measure_collapsed_vowels import count_collapsed_vowels
from matcha.utils.model import LOG_DURATION_OFFSET

# A boundary that moved by more than this many fine frames counts as moved (2 frames = 10.7ms).
MOVED_BOUNDARY_MIN_FRAMES = 2


def symbol_boundaries(durations: np.ndarray) -> np.ndarray:
    """End frame of every symbol except the last one, whose end is fixed by the recording length."""
    return np.cumsum(durations)[:-1]


def print_tracked_words(utterance, durations):
    parts = []
    for segment in word_segments(utterance, durations):
        start = segment.start_frame * FINE_FRAME_SECONDS
        end = (segment.start_frame + segment.frames) * FINE_FRAME_SECONDS
        parts.append(f"{segment.label} {start:.2f}-{end:.2f}")
    print(f"   {utterance.relative_path}: " + " | ".join(parts))


def main():
    parser = argparse.ArgumentParser(description="Follow the alignment across checkpoints on the same files.")
    parser.add_argument("-i", "--data-config", required=True, help="Path to data YAML, e.g. configs/data/corpus-24k.yaml")
    parser.add_argument("-c", "--checkpoints", required=True, nargs="+", help="In training order")
    parser.add_argument("--track", nargs="*", default=[], help="Recordings whose word placement is printed")
    parser.add_argument("--files_per_speaker", type=int, default=50)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    data_config = OmegaConf.load(args.data_config)
    mel_dir = Path(data_config.mel_dir)
    filelists = [Path(data_config.valid_filelist_path), Path(data_config.train_filelist_path)]

    utterances_by_speaker = read_filelist(Path(data_config.valid_filelist_path))
    sample = []
    for speaker in sorted(utterances_by_speaker):
        sample.extend(utterances_by_speaker[speaker][:args.files_per_speaker])
    tracked = [find_utterance(filelists, relative_path) for relative_path in args.track]
    mels = {u.relative_path: load_fine_mel(mel_dir, u.relative_path) for u in sample + tracked}

    epochs: List[int] = []
    durations_per_checkpoint: List[Dict[str, np.ndarray]] = []
    for checkpoint_path in args.checkpoints:
        loaded = load_encoder(checkpoint_path)
        durations_by_file = {}
        total_vowels = 0
        total_collapsed = 0
        duration_errors = []
        for utterance in sample:
            encoder_frames, predicted = run_encoder(loaded, utterance)
            durations = monotonic_alignment_durations(alignment_scores(encoder_frames, mels[utterance.relative_path]))
            durations_by_file[utterance.relative_path] = durations
            vowel_count, collapsed_count, _ = count_collapsed_vowels(utterance, durations)
            total_vowels += vowel_count
            total_collapsed += collapsed_count
            predicted_log = torch.log(predicted + LOG_DURATION_OFFSET).numpy()
            aligned_log = np.log(durations + LOG_DURATION_OFFSET)
            duration_errors.append(np.abs(predicted_log - aligned_log).mean())

        epochs.append(loaded.epoch)
        durations_per_checkpoint.append(durations_by_file)
        collapsed_share = 100 * total_collapsed / max(1, total_vowels)
        print(f"== epoch {loaded.epoch}  ({checkpoint_path})")
        print(f"   collapsed vowels {collapsed_share:.1f}%   duration error {np.mean(duration_errors):.3f}")
        for utterance in tracked:
            encoder_frames, _ = run_encoder(loaded, utterance)
            durations = monotonic_alignment_durations(alignment_scores(encoder_frames, mels[utterance.relative_path]))
            print_tracked_words(utterance, durations)

    print(f"\n{'from':>6} {'to':>6}   boundaries moved >{MOVED_BOUNDARY_MIN_FRAMES} frames per 100 epochs   mean shift")
    for index in range(1, len(epochs)):
        moved_shares = []
        mean_shifts = []
        for utterance in sample:
            before = symbol_boundaries(durations_per_checkpoint[index - 1][utterance.relative_path])
            after = symbol_boundaries(durations_per_checkpoint[index][utterance.relative_path])
            shift = np.abs(before - after)
            moved_shares.append((shift > MOVED_BOUNDARY_MIN_FRAMES).mean())
            mean_shifts.append(shift.mean())
        epoch_span = max(1, epochs[index] - epochs[index - 1])
        moved_per_100_epochs = 100 * np.mean(moved_shares) * 100 / epoch_span
        mean_shift_ms = np.mean(mean_shifts) * FINE_FRAME_SECONDS * 1000
        print(f"{epochs[index - 1]:6d} {epochs[index]:6d}   {moved_per_100_epochs:38.1f}%   {mean_shift_ms:6.1f} ms")


if __name__ == "__main__":
    main()
