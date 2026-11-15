"""
Measure how strongly Gaussian noise on the alignment scores moves the alignment, for a range of noise scales.

VITS2 adds noise to the score matrix before monotonic alignment search, during training, so the alignment can
leave a wrong path that the prior loss would otherwise keep reinforcing. The noise is a standard Gaussian times
the spread of the score matrix times a noise scale. The useful scale depends on how far apart our scores are,
so it cannot be copied from VITS2. This script reports, per checkpoint and per noise scale:
  - the noise per score cell, against the score gap between a frame's own symbol and its neighbour,
  - how many symbol boundaries the noise moves, compared with the alignment without noise,
  - the collapsed vowel share under noise.

For the pinned recordings it also answers: how much better does the current (wrong) path score than the path
that puts a known pause on the word gap where it belongs, and how likely is noise of each scale to flip that.
The pinned path is found by the same search, with the word gap forced onto the pause and nothing else onto it.

Runs on the GPU with the alignment kernel training uses.

Usage:
  python -m matcha.utils.measure_alignment_noise -i configs/data/corpus-24k.yaml -c <ckpt...> \
      --pin 4/1423:4:1.38:1.48 4/1428:1:1.19:1.42
  A pin is recording:word_index:pause_start_seconds:pause_end_seconds; the word gap after that word (0-based)
  is forced onto the pause.
"""

import argparse
import math
from pathlib import Path
from typing import List, NamedTuple

import numpy as np
import torch
from omegaconf import OmegaConf
from super_monotonic_align import maximum_path as maximum_path_gpu

from matcha.text.symbols import SPACE_ID
from matcha.utils.alignment_diagnostics import (
    FINE_FRAME_SECONDS,
    find_utterance,
    load_encoder,
    load_fine_mel,
    read_filelist,
    run_encoder,
    word_segments,
)
from matcha.utils.measure_collapsed_vowels import count_collapsed_vowels

NOISE_SCALES = [0.003, 0.01, 0.03, 0.1, 0.3]
NOISE_DRAWS = 2
MOVED_BOUNDARY_MIN_FRAMES = 2
DEVICE = "cuda"


class Pin(NamedTuple):
    relative_path: str
    word_index: int
    pause_start_frame: int
    pause_end_frame: int


def parse_pin(text: str) -> Pin:
    relative_path, word_index, start_seconds, end_seconds = text.split(":")
    start_frame = round(float(start_seconds) / FINE_FRAME_SECONDS)
    end_frame = round(float(end_seconds) / FINE_FRAME_SECONDS)
    return Pin(relative_path, int(word_index), start_frame, end_frame)


def score_matrix(encoder_frames: torch.Tensor, mel: torch.Tensor) -> torch.Tensor:
    """Same score as MatchaTTS.find_alignment: minus half the squared distance. shape: (n_symbols, n_frames)"""
    encoder_frames = encoder_frames.to(DEVICE)
    mel = mel.to(DEVICE)
    mel_energy = -0.5 * (mel ** 2).sum(dim=0, keepdim=True)
    cross_term = encoder_frames.T @ mel
    encoder_energy = -0.5 * (encoder_frames ** 2).sum(dim=0, keepdim=True).T
    return mel_energy + cross_term + encoder_energy


def search_durations(scores: torch.Tensor) -> np.ndarray:
    mask = torch.ones(scores.shape, dtype=torch.int32, device=DEVICE)[None]
    # The kernel overwrites its input with the running path scores, so it gets a copy.
    path = maximum_path_gpu(scores[None].clone(), mask, scores.dtype)
    return path[0].sum(-1).long().cpu().numpy()


def path_score(scores: torch.Tensor, durations: np.ndarray) -> float:
    symbol_per_frame = torch.from_numpy(np.repeat(np.arange(len(durations)), durations)).to(DEVICE)
    frames = torch.arange(scores.shape[1], device=DEVICE)
    return scores[symbol_per_frame, frames].sum().item()


def symbol_per_frame(durations: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(durations)), durations)


def boundaries(durations: np.ndarray) -> np.ndarray:
    return np.cumsum(durations)[:-1]


def neighbour_score_gap(scores: torch.Tensor, durations: np.ndarray) -> float:
    """Median, over frames, of how much better a frame's own symbol scores than the better of its two neighbours."""
    own = torch.from_numpy(symbol_per_frame(durations)).to(DEVICE)
    frames = torch.arange(scores.shape[1], device=DEVICE)
    last_symbol = scores.shape[0] - 1
    previous_symbol = (own - 1).clamp(min=0)
    next_symbol = (own + 1).clamp(max=last_symbol)
    own_score = scores[own, frames]
    best_neighbour = torch.maximum(scores[previous_symbol, frames], scores[next_symbol, frames])
    return (own_score - best_neighbour).median().item()


def noisy(scores: torch.Tensor, noise_scale: float, generator: torch.Generator) -> torch.Tensor:
    noise = torch.randn(scores.shape, device=DEVICE, generator=generator)
    return scores + noise * scores.std() * noise_scale


def pinned_durations(scores: torch.Tensor, gap_symbol: int, start_frame: int, end_frame: int) -> np.ndarray:
    """Best path in which the word gap holds exactly the pause frames and no other symbol touches them."""
    forbidden = torch.finfo(scores.dtype).min / 4
    pinned = scores.clone()
    pinned[:, start_frame:end_frame] = forbidden
    pinned[gap_symbol, :] = forbidden
    pinned[gap_symbol, start_frame:end_frame] = scores[gap_symbol, start_frame:end_frame]
    return search_durations(pinned)


def word_gap_after(phoneme_ids: List[int], word_index: int) -> int:
    """The leading space is the gap before word 0, so the gap after word w is the (w+1)-th space after it."""
    space_positions = [position for position, phoneme_id in enumerate(phoneme_ids) if phoneme_id == SPACE_ID]
    return space_positions[word_index + 1]


def words_text(utterance, durations) -> str:
    parts = []
    for segment in word_segments(utterance, durations):
        start = segment.start_frame * FINE_FRAME_SECONDS
        end = (segment.start_frame + segment.frames) * FINE_FRAME_SECONDS
        parts.append(f"{segment.label} {start:.2f}-{end:.2f}")
    return " | ".join(parts)


def report_sample(loaded, sample, mels, generator):
    stds = []
    gaps = []
    clean_durations = {}
    scores_by_file = {}
    clean_vowels = 0
    clean_collapsed = 0
    for utterance in sample:
        encoder_frames, _ = run_encoder(loaded, utterance)
        scores = score_matrix(encoder_frames, mels[utterance.relative_path])
        durations = search_durations(scores)
        scores_by_file[utterance.relative_path] = scores
        clean_durations[utterance.relative_path] = durations
        stds.append(scores.std().item())
        gaps.append(neighbour_score_gap(scores, durations))
        vowels, collapsed, _ = count_collapsed_vowels(utterance, durations)
        clean_vowels += vowels
        clean_collapsed += collapsed

    print(f"   score spread (std of the matrix) median {np.median(stds):.2f}   "
          f"own-vs-neighbour score gap median {np.median(gaps):.3f}   "
          f"collapsed vowels without noise {100 * clean_collapsed / clean_vowels:.2f}%")
    print(f"   {'scale':>6} {'noise/cell':>10} {'moved>=1':>9} {'moved>2':>8} {'mean shift':>10} {'collapsed':>9}")
    for noise_scale in NOISE_SCALES:
        moved_any = []
        moved_far = []
        shifts = []
        vowels_total = 0
        collapsed_total = 0
        for utterance in sample:
            scores = scores_by_file[utterance.relative_path]
            before = boundaries(clean_durations[utterance.relative_path])
            for _ in range(NOISE_DRAWS):
                durations = search_durations(noisy(scores, noise_scale, generator))
                shift = np.abs(boundaries(durations) - before)
                moved_any.append((shift >= 1).mean())
                moved_far.append((shift > MOVED_BOUNDARY_MIN_FRAMES).mean())
                shifts.append(shift.mean())
                vowels, collapsed, _ = count_collapsed_vowels(utterance, durations)
                vowels_total += vowels
                collapsed_total += collapsed
        noise_per_cell = noise_scale * np.median(stds)
        mean_shift_ms = np.mean(shifts) * FINE_FRAME_SECONDS * 1000
        print(f"   {noise_scale:6.3f} {noise_per_cell:10.3f} {100 * np.mean(moved_any):8.1f}% "
              f"{100 * np.mean(moved_far):7.1f}% {mean_shift_ms:8.1f}ms {100 * collapsed_total / vowels_total:8.2f}%")


def report_pin(loaded, utterance, mel, pin: Pin):
    encoder_frames, _ = run_encoder(loaded, utterance)
    scores = score_matrix(encoder_frames, mel)
    free = search_durations(scores)
    gap_symbol = word_gap_after(utterance.phoneme_ids, pin.word_index)
    pinned = pinned_durations(scores, gap_symbol, pin.pause_start_frame, pin.pause_end_frame)
    margin = path_score(scores, free) - path_score(scores, pinned)
    differing_frames = int((symbol_per_frame(free) != symbol_per_frame(pinned)).sum())
    spread = scores.std().item()

    print(f"   {pin.relative_path}: current path scores {margin:.1f} better than the pinned one, "
          f"{differing_frames} frames assigned differently")
    print(f"      current: {words_text(utterance, free)}")
    print(f"      pinned:  {words_text(utterance, pinned)}")
    # The two paths differ on `differing_frames` frames; on each of them the noise difference between the two
    # cells is Gaussian with twice the cell variance, so the total noise difference has std cell_std * sqrt(2 D).
    chances = []
    for noise_scale in NOISE_SCALES:
        noise_difference_std = noise_scale * spread * math.sqrt(2 * max(1, differing_frames))
        flip_chance = 0.5 * math.erfc(margin / noise_difference_std / math.sqrt(2))
        chances.append(f"{noise_scale}: {100 * flip_chance:.2g}%")
    print(f"      chance noise prefers the pinned path over the current one, per step: {'   '.join(chances)}")


def main():
    parser = argparse.ArgumentParser(description="Measure how noise on the alignment scores moves the alignment.")
    parser.add_argument("-i", "--data-config", required=True)
    parser.add_argument("-c", "--checkpoints", required=True, nargs="+")
    parser.add_argument("--pin", nargs="*", default=[])
    parser.add_argument("--files_per_speaker", type=int, default=20)
    parser.add_argument("--exclude_speakers", type=int, nargs="*", default=[11])
    args = parser.parse_args()

    data_config = OmegaConf.load(args.data_config)
    mel_dir = Path(data_config.mel_dir)
    filelists = [Path(data_config.valid_filelist_path), Path(data_config.train_filelist_path)]
    utterances_by_speaker = read_filelist(Path(data_config.train_filelist_path))
    sample = []
    for speaker in sorted(utterances_by_speaker):
        if speaker in args.exclude_speakers:
            continue
        sample.extend(utterances_by_speaker[speaker][:args.files_per_speaker])
    pins = [parse_pin(text) for text in args.pin]
    pinned_utterances = [find_utterance(filelists, pin.relative_path) for pin in pins]
    mels = {u.relative_path: load_fine_mel(mel_dir, u.relative_path) for u in sample + pinned_utterances}

    for checkpoint_path in args.checkpoints:
        loaded = load_encoder(checkpoint_path)
        generator = torch.Generator(device=DEVICE).manual_seed(0)
        print(f"== epoch {loaded.epoch}  ({checkpoint_path}), {len(sample)} train files")
        report_sample(loaded, sample, mels, generator)
        for utterance, pin in zip(pinned_utterances, pins):
            report_pin(loaded, utterance, mels[pin.relative_path], pin)


if __name__ == "__main__":
    main()
