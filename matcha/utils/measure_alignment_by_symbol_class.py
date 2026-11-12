"""
Measure whether alignment errors concentrate on particular kinds of symbols in the phonemization scheme.

Every voiced phoneme is written as three symbols (opening separator, phoneme, closing separator). eSpeak also
emits symbols that carry no sound of their own but still get at least one fine frame from the alignment:
stress marks, written before the opening separator of a stressed vowel group, and length marks, written after
the closing separator of a long vowel. Diphthongs are written as two vowel groups in a row.

The script answers three questions:
1. How many frames do length marks and stress marks get, and does the length mark carry the steady part of
   its long vowel instead of the vowel group itself?
2. Do short vowels, long vowels and diphthong halves, and stressed and unstressed vowels, collapse or swell
   at different rates?
3. Inside swollen phoneme groups, do the extra frames sit on the separators or on the phoneme?

A phoneme group is swollen when it gets more than SWOLLEN_FACTOR times the median length of the same phoneme
for the same speaker. The median comes from the same alignments, so it is a reference for "typical", not a
ground truth.

The alignment is computed the same way training computes it, on the CPU, so the script can run while training
holds the GPU. Alignments are cached, so the report can be recomputed without running the Encoder again.

Usage:
  python -m matcha.utils.measure_alignment_by_symbol_class -i configs/data/corpus-24k.yaml \
      -c logs/train/v24/checkpoint_epoch=789.ckpt --split train --files_per_speaker 150 \
      --alignment_cache /tmp/alignments_789_train.pkl
"""

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple

import numpy as np
import torch
from omegaconf import OmegaConf

from matcha.text.symbols import PRE_ID, POST_ID, SPACE_ID, symbol_to_id, symbols, vowels
from matcha.utils.alignment_diagnostics import (
    FINE_FRAME_SECONDS,
    ground_truth_alignment,
    load_encoder,
    load_fine_mel,
    read_filelist,
)

# Same threshold as measure_collapsed_vowels: 3 fine frames are 16ms, shorter than any real vowel.
COLLAPSED_GROUP_MAX_FRAMES = 3
SWOLLEN_FACTOR = 2.5
# Phonemes seen fewer times than this for a speaker have no reliable median length, so they are not judged.
MIN_OCCURRENCES_FOR_MEDIAN = 10

LENGTH_MARK_ID = symbol_to_id["ː"]
STRESS_MARK_IDS = {symbol_to_id["ˈ"], symbol_to_id["ˌ"]}


class AlignedUtterance(NamedTuple):
    relative_path: str
    speaker: int
    phoneme_ids: List[int]
    durations: np.ndarray


class PhonemeGroup(NamedTuple):
    relative_path: str
    speaker: int
    phoneme: str
    is_vowel: bool
    vowel_kind: str
    is_stressed: bool
    opening_frames: int
    core_frames: int
    closing_frames: int
    length_mark_frames: int

    @property
    def group_frames(self) -> int:
        return self.opening_frames + self.core_frames + self.closing_frames

    @property
    def sound_frames(self) -> int:
        """The group plus its length mark: everything that holds the sound of a long vowel."""
        return self.group_frames + self.length_mark_frames


def is_opening_separator(phoneme_id: int) -> bool:
    return PRE_ID <= phoneme_id < POST_ID


def is_vowel_group_start(phoneme_ids: List[int], position: int) -> bool:
    is_inside_sequence = 0 <= position < len(phoneme_ids)
    if not is_inside_sequence:
        return False
    phoneme_id = phoneme_ids[position]
    return is_opening_separator(phoneme_id) and symbols[phoneme_id - PRE_ID] in vowels


def classify_vowel(phoneme_ids: List[int], opening_position: int) -> str:
    # A vowel group directly touching another vowel group is half of a diphthong; eSpeak writes no symbol between.
    touches_previous_vowel_group = is_vowel_group_start(phoneme_ids, opening_position - 3)
    touches_next_vowel_group = is_vowel_group_start(phoneme_ids, opening_position + 3)
    if touches_previous_vowel_group or touches_next_vowel_group:
        return "diphthong half"
    after_group = opening_position + 3
    is_followed_by_length_mark = after_group < len(phoneme_ids) and phoneme_ids[after_group] == LENGTH_MARK_ID
    if is_followed_by_length_mark:
        return "long"
    return "short"


def extract_groups(utterance: AlignedUtterance) -> List[PhonemeGroup]:
    phoneme_ids = utterance.phoneme_ids
    durations = utterance.durations
    groups = []
    for position, phoneme_id in enumerate(phoneme_ids):
        if not is_opening_separator(phoneme_id):
            continue
        phoneme = symbols[phoneme_id - PRE_ID]
        is_vowel = phoneme in vowels
        vowel_kind = "-"
        is_stressed = False
        length_mark_frames = 0
        if is_vowel:
            vowel_kind = classify_vowel(phoneme_ids, position)
            is_stressed = position > 0 and phoneme_ids[position - 1] in STRESS_MARK_IDS
            if vowel_kind == "long":
                length_mark_frames = int(durations[position + 3])
        groups.append(
            PhonemeGroup(
                relative_path=utterance.relative_path,
                speaker=utterance.speaker,
                phoneme=phoneme,
                is_vowel=is_vowel,
                vowel_kind=vowel_kind,
                is_stressed=is_stressed,
                opening_frames=int(durations[position]),
                core_frames=int(durations[position + 1]),
                closing_frames=int(durations[position + 2]),
                length_mark_frames=length_mark_frames,
            )
        )
    return groups


def compute_alignments(args) -> List[AlignedUtterance]:
    data_config = OmegaConf.load(args.data_config)
    if args.split == "validate":
        filelist_path = Path(data_config.valid_filelist_path)
    else:
        filelist_path = Path(data_config.train_filelist_path)
    mel_dir = Path(data_config.mel_dir)

    loaded = load_encoder(args.checkpoint)
    print(f"Checkpoint: {args.checkpoint} (epoch {loaded.epoch}), split: {args.split}", flush=True)
    utterances_by_speaker = read_filelist(filelist_path)

    aligned = []
    for speaker in sorted(utterances_by_speaker):
        if speaker in args.exclude_speakers:
            continue
        utterances = utterances_by_speaker[speaker][:args.files_per_speaker]
        for utterance in utterances:
            mel = load_fine_mel(mel_dir, utterance.relative_path)
            durations = ground_truth_alignment(loaded, utterance, mel)
            aligned.append(AlignedUtterance(utterance.relative_path, speaker, utterance.phoneme_ids, durations))
        print(f"  aligned speaker {speaker}: {len(utterances)} files", flush=True)
    return aligned


def load_or_compute_alignments(args) -> List[AlignedUtterance]:
    has_cache = args.alignment_cache is not None and Path(args.alignment_cache).exists()
    if has_cache:
        print(f"Reading alignments from {args.alignment_cache}")
        with open(args.alignment_cache, "rb") as f:
            return pickle.load(f)
    aligned = compute_alignments(args)
    if args.alignment_cache is not None:
        with open(args.alignment_cache, "wb") as f:
            pickle.dump(aligned, f)
    return aligned


def median_frames_by_speaker_and_phoneme(groups: List[PhonemeGroup]) -> Dict[tuple, float]:
    lengths = defaultdict(list)
    for group in groups:
        lengths[(group.speaker, group.phoneme, group.vowel_kind)].append(group.sound_frames)
    return {
        key: float(np.median(values)) for key, values in lengths.items() if len(values) >= MIN_OCCURRENCES_FOR_MEDIAN
    }


def is_swollen(group: PhonemeGroup, medians: Dict[tuple, float]) -> bool:
    median = medians.get((group.speaker, group.phoneme, group.vowel_kind))
    if median is None:
        return False
    return group.sound_frames > SWOLLEN_FACTOR * median


def frames_in_ms(frames: float) -> float:
    return frames * FINE_FRAME_SECONDS * 1000


def silent_mark_label(phoneme_ids: List[int], position: int, names: Dict[str, str]) -> str:
    phoneme_id = phoneme_ids[position]
    label = names[symbols[phoneme_id]]
    if phoneme_id not in STRESS_MARK_IDS:
        return label
    # A stress mark after a word gap starts the word; anywhere else it sits inside a transition between two groups.
    follows_word_gap = position > 0 and phoneme_ids[position - 1] == SPACE_ID
    if follows_word_gap:
        return label + ", after word gap"
    return label + ", inside word"


def report_silent_marks(aligned: List[AlignedUtterance], groups: List[PhonemeGroup]):
    print("\n1. Symbols with no sound of their own")
    print(f"{'symbol':<36} {'count':>7} {'median ms':>9} {'mean ms':>8} {'at 1 frame':>10} {'>= 27 ms':>9}")
    frames_by_mark = defaultdict(list)
    names = {"ˈ": "primary stress ˈ", "ˌ": "secondary stress ˌ", "ː": "length mark ː"}
    for utterance in aligned:
        for position, phoneme_id in enumerate(utterance.phoneme_ids):
            is_silent_mark = phoneme_id in STRESS_MARK_IDS or phoneme_id == LENGTH_MARK_ID
            if not is_silent_mark:
                continue
            frames_by_mark[silent_mark_label(utterance.phoneme_ids, position, names)].append(
                int(utterance.durations[position])
            )
    for label in sorted(frames_by_mark):
        frames = np.array(frames_by_mark[label])
        at_one_frame = 100 * np.mean(frames == 1)
        at_least_five_frames = 100 * np.mean(frames >= 5)
        print(
            f"{label:<36} {len(frames):7d} {frames_in_ms(np.median(frames)):9.1f} "
            f"{frames_in_ms(frames.mean()):8.1f} {at_one_frame:9.1f}% {at_least_five_frames:8.1f}%"
        )

    long_vowels = [group for group in groups if group.vowel_kind == "long"]
    length_mark_share = np.array([group.length_mark_frames / group.sound_frames for group in long_vowels])
    mark_longer_than_core = np.mean([group.length_mark_frames > group.core_frames for group in long_vowels])
    print(f"\nLong vowels: {len(long_vowels)}")
    print(f"  share of the long vowel's frames on the length mark: median {100 * np.median(length_mark_share):.0f}%, "
          f"mean {100 * length_mark_share.mean():.0f}%")
    print(f"  length mark gets more frames than the vowel itself: {100 * mark_longer_than_core:.1f}%")
    print(f"  length mark holds over half of the long vowel: {100 * np.mean(length_mark_share > 0.5):.1f}%")


def collapse_and_swell_rates(selected: List[PhonemeGroup], medians: Dict[tuple, float]) -> str:
    if not selected:
        return "no groups"
    collapsed = np.mean([group.group_frames <= COLLAPSED_GROUP_MAX_FRAMES for group in selected])
    swollen = np.mean([is_swollen(group, medians) for group in selected])
    median_ms = frames_in_ms(np.median([group.sound_frames for group in selected]))
    return f"{len(selected):7d} {median_ms:9.1f} {100 * collapsed:9.2f}% {100 * swollen:8.2f}%"


def report_vowel_kinds(groups: List[PhonemeGroup], medians: Dict[tuple, float]):
    print(f"\n2. Collapsed (group of three symbols <= {COLLAPSED_GROUP_MAX_FRAMES} frames) and swollen "
          f"(> {SWOLLEN_FACTOR} x the speaker's median for that phoneme) groups")
    print(f"{'kind':<28} {'count':>7} {'median ms':>9} {'collapsed':>10} {'swollen':>9}")
    vowel_groups = [group for group in groups if group.is_vowel]
    for kind in ("short", "long", "diphthong half"):
        for is_stressed in (True, False):
            selected = [g for g in vowel_groups if g.vowel_kind == kind and g.is_stressed == is_stressed]
            if is_stressed:
                label = f"vowel, {kind}, stressed"
            else:
                label = f"vowel, {kind}, unstressed"
            print(f"{label:<28} {collapse_and_swell_rates(selected, medians)}")
    consonant_groups = [group for group in groups if not group.is_vowel]
    print(f"{'consonant':<28} {collapse_and_swell_rates(consonant_groups, medians)}")


def separator_shares(selected: List[PhonemeGroup]) -> str:
    opening = sum(group.opening_frames for group in selected)
    core = sum(group.core_frames for group in selected)
    closing = sum(group.closing_frames for group in selected)
    length_mark = sum(group.length_mark_frames for group in selected)
    total = max(1, opening + core + closing + length_mark)
    return (f"{len(selected):7d} {100 * opening / total:8.0f}% {100 * core / total:8.0f}% "
            f"{100 * closing / total:8.0f}% {100 * length_mark / total:8.0f}%")


def report_where_swollen_frames_go(groups: List[PhonemeGroup], medians: Dict[tuple, float]):
    print("\n3. Where the frames sit inside a group: swollen groups compared to all other groups")
    print(f"{'groups':<34} {'count':>7} {'opening':>9} {'phoneme':>9} {'closing':>9} {'length':>9}")
    categories = (
        ("consonant", lambda g: not g.is_vowel),
        ("vowel, short", lambda g: g.vowel_kind == "short"),
        ("vowel, long", lambda g: g.vowel_kind == "long"),
        ("vowel, diphthong half", lambda g: g.vowel_kind == "diphthong half"),
    )
    for name, belongs in categories:
        members = [group for group in groups if belongs(group)]
        swollen = [group for group in members if is_swollen(group, medians)]
        normal = [group for group in members if not is_swollen(group, medians)]
        print(f"{name + ', swollen':<34} {separator_shares(swollen)}")
        print(f"{name + ', other':<34} {separator_shares(normal)}")


def report_most_swollen_phonemes(groups: List[PhonemeGroup], medians: Dict[tuple, float]):
    print("\nPhonemes that swell most often (at least 200 occurrences)")
    by_phoneme = defaultdict(list)
    for group in groups:
        by_phoneme[(group.phoneme, group.vowel_kind)].append(is_swollen(group, medians))
    rates = [(np.mean(flags), len(flags), key) for key, flags in by_phoneme.items() if len(flags) >= 200]
    rates.sort(reverse=True)
    for rate, count, (phoneme, kind) in rates[:12]:
        print(f"  {phoneme} ({kind}): {100 * rate:.2f}% of {count}")


def main():
    parser = argparse.ArgumentParser(description="Alignment errors by symbol class.")
    parser.add_argument("-i", "--data-config", required=True)
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("--split", choices=["validate", "train"], default="train")
    parser.add_argument("--files_per_speaker", type=int, default=150)
    # Speaker 11 has noise-gated digital silence inside sentences, which distorts every alignment statistic.
    parser.add_argument("--exclude_speakers", type=int, nargs="*", default=[11])
    parser.add_argument("--threads", type=int, default=8, help="CPU threads for the Encoder forward pass")
    parser.add_argument("--alignment_cache", default=None, help="Pickle file to reuse alignments between runs")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    aligned = load_or_compute_alignments(args)
    groups = []
    for utterance in aligned:
        groups.extend(extract_groups(utterance))
    medians = median_frames_by_speaker_and_phoneme(groups)

    print(f"\n{len(aligned)} files, {len(groups)} phoneme groups, speakers excluded: {args.exclude_speakers}")
    report_silent_marks(aligned, groups)
    report_vowel_kinds(groups, medians)
    report_where_swollen_frames_go(groups, medians)
    report_most_swollen_phonemes(groups, medians)


if __name__ == "__main__":
    main()
