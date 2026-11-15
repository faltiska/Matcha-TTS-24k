"""
Measures where the alignment search (MAS) puts each pause of a recording, and how expensive each choice was.

A pause is a run of silent fine frames inside the speech. To know where a pause really is, MAS is run a second
time on the mel with every pause cut out: without silence the path is decided by the speech alone, and the cut
point falls between the two symbols that really surround the pause (the reference position). The normal MAS
result is then compared with it:
  on its gap         most pause frames go to the symbols between the two words around the reference position
  neighbour phoneme  most pause frames go to the phoneme right next to those symbols
  displaced          most pause frames go further away; the symbols in between get squeezed
  inside a word      the reference position is inside a word (stop closures, or the reference is wrong)

For every pause it also reports the cost per pause frame of putting the pause on the right gap, and on the
cheapest other symbol within reach (the competitor). A pause is displaced when the gap's Encoder frame does not
look like silence and a silence-like competitor is close.

Reads the matcha wavs and fine mels, so times refer to data/corpus-24k/wav. Runs on the CPU.

Usage:
  python -m matcha.utils.measure_pause_placement -i configs/data/corpus-24k-v25.yaml -c <ckpt> [--files_per_speaker 150]
  python -m matcha.utils.measure_pause_placement -i configs/data/corpus-24k-v25.yaml -c <ckpt> -u 4/1423 4/1428
"""

import argparse
import random
import re
import wave
from collections import Counter
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from omegaconf import OmegaConf

from matcha.text.symbols import SPACE_ID, symbols, voiced_phoneme_ids
from matcha.utils.alignment_diagnostics import (
    FINE_FRAME_SECONDS,
    FINE_HOP_LENGTH,
    WORD_GAP_LABEL,
    Utterance,
    alignment_scores,
    find_utterance,
    load_encoder,
    load_fine_mel,
    monotonic_alignment_durations,
    read_filelist,
    run_encoder,
    word_segments,
)

# A fine frame louder than this counts as speech when finding where speech starts and ends.
SPEECH_THRESHOLD_DB = -45.0
# A fine frame quieter than this counts as silence.
SILENCE_THRESHOLD_DB = -50.0
# 15 fine frames are 80 ms; shorter silences are mostly stop closures.
MIN_PAUSE_FRAMES = 15
# How many symbols to each side of the reference position are searched for the cheapest competitor.
COMPETITOR_REACH = 20
PUNCTUATION = set(",;:.!?—…-")


class PausePlacement(NamedTuple):
    audio: str
    speaker: int
    start_seconds: float
    milliseconds: float
    outcome: str
    has_punctuation: Optional[bool]  # None when the reference position is inside a word
    gap_cost: float
    competitor_cost: float
    competitor: str
    competitor_distance: int
    chosen_symbol: str
    words_moved: int
    context: str  # the text around the reference position, with ⟨⟩ where the pause really is
    # MAS gave the pause to a gap with written punctuation. Next to a short word ("Max. I want") the reference cut
    # point can land one word late, so such a pause is most likely placed right and the reference is wrong.
    chosen_on_punctuation: bool
    # Index, in the transcript's words, of the word right before the pause; -1 when the pause comes before every word.
    word_before: int
    # The phonemized text has as many words as the transcript, so word_before points at the right transcript word.
    words_match_text: bool


def label(phoneme_id: int) -> str:
    if phoneme_id == SPACE_ID:
        return WORD_GAP_LABEL
    return symbols[phoneme_id]


def frame_levels_db(wav_path: Path, n_frames: int) -> np.ndarray:
    """Level of every fine frame, in dB. Frames past the end of the wav count as silent."""
    with wave.open(str(wav_path)) as w:
        samples = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    frame_count = len(samples) // FINE_HOP_LENGTH
    frames = samples[:frame_count * FINE_HOP_LENGTH].reshape(frame_count, FINE_HOP_LENGTH)
    levels = 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)
    padded = np.full(n_frames, -120.0)
    used = min(frame_count, n_frames)
    padded[:used] = levels[:used]
    return padded


def find_pauses(levels_db: np.ndarray, silence_db: float = SILENCE_THRESHOLD_DB,
                min_frames: int = MIN_PAUSE_FRAMES) -> List[Tuple[int, int]]:
    """First and one-past-last fine frame of every pause between the first and the last speech frame."""
    speech_frames = np.where(levels_db > SPEECH_THRESHOLD_DB)[0]
    if len(speech_frames) == 0:
        return []
    pauses = []
    pause_start = None
    for frame in range(speech_frames[0], speech_frames[-1] + 1):
        is_silent = levels_db[frame] < silence_db
        if is_silent and pause_start is None:
            pause_start = frame
        if not is_silent and pause_start is not None:
            is_long_enough = frame - pause_start >= min_frames
            if is_long_enough:
                pauses.append((pause_start, frame))
            pause_start = None
    return pauses


def word_index_per_symbol(phoneme_ids: List[int]) -> np.ndarray:
    """Index of the word each symbol belongs to; word gaps get -1."""
    word_of = np.full(len(phoneme_ids), -1)
    word = -1
    inside_word = False
    for position, phoneme_id in enumerate(phoneme_ids):
        if phoneme_id == SPACE_ID:
            inside_word = False
            continue
        if not inside_word:
            word += 1
            inside_word = True
        word_of[position] = word
    return word_of


def gap_zones(phoneme_ids: List[int]) -> List[Tuple[int, int]]:
    """
    For every word gap, the first and last position of the symbols between the last voiced phoneme of one word and
    the first voiced phoneme of the next: separators, annotations, punctuation and the space itself.
    """
    voiced_positions = [p for p, i in enumerate(phoneme_ids) if i in voiced_phoneme_ids]
    zones = []
    for previous, following in zip(voiced_positions, voiced_positions[1:]):
        contains_space = SPACE_ID in phoneme_ids[previous + 1:following]
        if contains_space:
            zones.append((previous + 1, following - 1))
    return zones


def symbols_per_frame(durations: np.ndarray) -> np.ndarray:
    return np.repeat(np.arange(len(durations)), durations)


def mas_without_pauses(scores: np.ndarray, pauses: List[Tuple[int, int]]) -> Tuple[Optional[np.ndarray], np.ndarray]:
    """MAS durations on the frames left after cutting out every pause, and the mask of those frames."""
    kept = np.ones(scores.shape[1], dtype=bool)
    for start, end in pauses:
        kept[start:end] = False
    too_few_frames_left = kept.sum() < scores.shape[0]
    if too_few_frames_left:
        return None, kept
    return monotonic_alignment_durations(scores[:, kept].copy()), kept


def zone_around(zones, symbol_before: int, symbol_after: int) -> Optional[Tuple[int, int]]:
    for first, last in zones:
        before_is_at_zone = first - 1 <= symbol_before <= last
        after_is_at_zone = first <= symbol_after <= last + 1
        if before_is_at_zone and after_is_at_zone:
            return first, last
    return None


def place_pause(utterance, scores, chosen, reference, kept, zones, word_of, text_words, pause) -> PausePlacement:
    ids = utterance.phoneme_ids
    start, end = pause
    kept_frames_before = int(kept[:start].sum())
    symbol_before = int(reference[kept_frames_before - 1])
    symbol_after = int(reference[min(kept_frames_before, len(reference) - 1)])
    zone = zone_around(zones, symbol_before, symbol_after)

    pause_owners = chosen[start:end]
    chosen_position = int(np.bincount(pause_owners).argmax())
    # Average cost per pause frame of giving the pause to each symbol.
    cost = -scores[:, start:end].mean(axis=1)

    if zone is None:
        outcome = "inside a word"
        has_punctuation = None
        first, last = symbol_before, symbol_after
    else:
        first, last = zone
        has_punctuation = any(symbols[ids[p]] in PUNCTUATION for p in range(first, last + 1))
        share_on_gap = float(((pause_owners >= first) & (pause_owners <= last)).mean())
        is_next_to_gap = first - 1 <= chosen_position <= last + 1
        if share_on_gap >= 0.5:
            outcome = "on its gap"
        elif is_next_to_gap:
            outcome = "neighbour phoneme"
        else:
            outcome = "displaced"

    reach = range(max(0, first - COMPETITOR_REACH), min(len(ids), last + 1 + COMPETITOR_REACH))
    competitors = [p for p in reach if not first - 1 <= p <= last + 1]
    competitor = first
    if competitors:
        competitor = min(competitors, key=lambda p: cost[p])

    words_moved = 0
    both_in_words = word_of[chosen_position] >= 0 and word_of[first] >= 0
    if both_in_words:
        words_moved = abs(int(word_of[chosen_position]) - int(word_of[first]))

    phonemized_word_before = word_before_pause(word_of, first)
    # When the reference puts the pause inside a word, the cut almost always fell into the word after the pause
    # (checked against a hand-corrected transcript: 262 of 297 one word late), so the pause follows the word before.
    is_reference_inside_word = zone is None
    if is_reference_inside_word and phonemized_word_before >= 1:
        phonemized_word_before -= 1
    text_word_before = phonemized_word_before
    if text_words is not None and phonemized_word_before >= 0:
        text_word_before = text_words[phonemized_word_before]

    return PausePlacement(
        audio=utterance.relative_path,
        speaker=utterance.speaker,
        start_seconds=start * FINE_FRAME_SECONDS,
        milliseconds=(end - start) * FINE_FRAME_SECONDS * 1000,
        outcome=outcome,
        has_punctuation=has_punctuation,
        gap_cost=float(cost[first:last + 1].min()),
        competitor_cost=float(cost[competitor]),
        competitor=label(ids[competitor]),
        competitor_distance=abs(competitor - first),
        chosen_symbol=label(ids[chosen_position]),
        words_moved=words_moved,
        context=pause_context(utterance.text, text_word_before),
        chosen_on_punctuation=is_on_punctuated_gap(ids, zones, chosen_position),
        word_before=text_word_before,
        words_match_text=text_words is not None,
    )


def is_on_punctuated_gap(phoneme_ids: List[int], zones, position: int) -> bool:
    for zone_first, zone_last in zones:
        is_in_zone = zone_first - 1 <= position <= zone_last + 1
        if is_in_zone:
            return any(symbols[phoneme_ids[p]] in PUNCTUATION for p in range(zone_first, zone_last + 1))
    return False


def word_before_pause(word_of: np.ndarray, first_gap_symbol: int) -> int:
    """The word of the last symbol before the gap, or -1 when no word comes before it."""
    earlier_words = [w for w in word_of[:first_gap_symbol] if w >= 0]
    if earlier_words:
        return int(earlier_words[-1])
    return -1


def pause_context(text: str, last_word_before: int) -> str:
    """Up to three words on each side of the pause."""
    words = text.split()
    before = " ".join(words[max(0, last_word_before - 2):last_word_before + 1])
    after = " ".join(words[last_word_before + 1:last_word_before + 4])
    return f"{before} ⟨⟩ {after}"


# Language of every recording read so far; the shared Utterance type does not keep it, but phonemizing single
# words needs it.
LANGUAGE_BY_AUDIO: Dict[str, str] = {}


def last_text_word_per_phonemized_word(utterance: Utterance, word_of: np.ndarray) -> Optional[List[int]]:
    """
    For every word of the phonemized text, the index of the last transcript word it covers.

    eSpeak joins short words ("of the" is phonemized as one word), so the phonemized text can have fewer words than
    the transcript. Each transcript word is then phonemized on its own, and consecutive transcript words are merged
    while that brings their phoneme count closer to the joined word's. Returns None when the phonemized text has
    more words than the transcript (numbers expanded by the text normalizer) or the merge does not add up.
    """
    text_word_count = len(utterance.text.split())
    phonemized_word_count = int(word_of.max()) + 1
    if phonemized_word_count == text_word_count:
        return list(range(text_word_count))
    language = LANGUAGE_BY_AUDIO.get(utterance.relative_path)
    if phonemized_word_count > text_word_count or language is None:
        return None
    from matcha.text.phonemizers import multilingual_phonemizer

    phonemized_lengths = [int(((word_of == w) & is_voiced(utterance.phoneme_ids)).sum()) for w in range(phonemized_word_count)]
    text_lengths = []
    for word in utterance.text.split():
        _, word_ids = multilingual_phonemizer(word, language)
        text_lengths.append(int(is_voiced(list(word_ids)).sum()))

    last_text_word = []
    text_index = 0
    for phonemized_index, target in enumerate(phonemized_lengths):
        covered = text_lengths[text_index]
        text_index += 1
        phonemized_words_left = phonemized_word_count - phonemized_index - 1
        while text_index < text_word_count and text_word_count - text_index > phonemized_words_left:
            merging_gets_closer = abs(covered + text_lengths[text_index] - target) < abs(covered - target)
            must_merge = phonemized_words_left == 0
            if not (merging_gets_closer or must_merge):
                break
            covered += text_lengths[text_index]
            text_index += 1
        last_text_word.append(text_index - 1)
    if text_index != text_word_count:
        return None
    return last_text_word


def is_voiced(phoneme_ids: List[int]) -> np.ndarray:
    return np.array([i in voiced_phoneme_ids for i in phoneme_ids])


def measure_utterance(loaded, utterance: Utterance, mel_dir: Path, wav_dir: Path,
                      silence_db: float = SILENCE_THRESHOLD_DB, min_frames: int = MIN_PAUSE_FRAMES):
    """Returns the placement of every pause, plus the MAS durations with and without the pauses."""
    mel = load_fine_mel(mel_dir, utterance.relative_path)
    levels_db = frame_levels_db(wav_dir / f"{utterance.relative_path}.wav", mel.shape[1])
    pauses = find_pauses(levels_db, silence_db, min_frames)
    if not pauses:
        return [], None, None, None
    encoder_frames, _ = run_encoder(loaded, utterance)
    scores = alignment_scores(encoder_frames, mel)
    chosen_durations = monotonic_alignment_durations(scores.copy())
    reference_durations, kept = mas_without_pauses(scores, pauses)
    if reference_durations is None:
        return [], None, None, None
    chosen = symbols_per_frame(chosen_durations)
    reference = symbols_per_frame(reference_durations)
    zones = gap_zones(utterance.phoneme_ids)
    word_of = word_index_per_symbol(utterance.phoneme_ids)
    text_words = last_text_word_per_phonemized_word(utterance, word_of)
    placements = [place_pause(utterance, scores, chosen, reference, kept, zones, word_of, text_words, p) for p in pauses]
    return placements, chosen_durations, reference_durations, kept


def words_text(utterance, durations, real_frames) -> str:
    """Word placement in seconds; real_frames maps the frames MAS saw back to frames of the recording."""
    parts = []
    for segment in word_segments(utterance, durations):
        is_empty_or_gap = segment.frames == 0 or segment.label == WORD_GAP_LABEL
        if is_empty_or_gap:
            continue
        start = real_frames[segment.start_frame] * FINE_FRAME_SECONDS
        end = (real_frames[segment.start_frame + segment.frames - 1] + 1) * FINE_FRAME_SECONDS
        parts.append(f"{segment.label} {start:.2f}-{end:.2f}")
    return " | ".join(parts)


def report_utterances(loaded, paths, filelists, mel_dir, wav_dir):
    for path in paths:
        utterance = find_utterance(filelists, path)
        placements, chosen, reference, kept = measure_utterance(loaded, utterance, mel_dir, wav_dir)
        print(f"\n{path} '{utterance.text}' (epoch {loaded.epoch})")
        if not placements:
            print("  no pause")
            continue
        all_frames = np.arange(len(kept))
        print(f"  MAS:            {words_text(utterance, chosen, all_frames)}")
        print(f"  without pauses: {words_text(utterance, reference, np.where(kept)[0])}")
        for p in placements:
            print(f"  pause {p.start_seconds:.2f} s, {p.milliseconds:.0f} ms: {p.outcome}, given to '{p.chosen_symbol}'; "
                  f"cost per frame on the gap {p.gap_cost:.1f}, competitor '{p.competitor}' {p.competitor_cost:.1f}")


def report_corpus(placements: List[PausePlacement], files_per_speaker: int, epoch: int):
    print(f"epoch {epoch}: {len(placements)} pauses of {MIN_PAUSE_FRAMES * FINE_FRAME_SECONDS * 1000:.0f} ms or more "
          f"in {files_per_speaker} training files per speaker")
    print("\nOutcome by what the text has at the reference position:")
    kinds = [(True, "written punctuation"), (False, "space only"), (None, "inside a word")]
    for has_punctuation, name in kinds:
        selected = [p for p in placements if p.has_punctuation is has_punctuation]
        counts = Counter(p.outcome for p in selected)
        print(f"  {name:20} {len(selected):5d}  " + "  ".join(f"{k}: {v}" for k, v in counts.most_common()))

    print("\nSpace-only pauses per speaker: on its gap / neighbour phoneme / displaced")
    for speaker in sorted({p.speaker for p in placements}):
        counts = Counter(p.outcome for p in placements if p.speaker == speaker and p.has_punctuation is False)
        print(f"  speaker {speaker:2d}: {counts['on its gap']:4d} / {counts['neighbour phoneme']:4d} / {counts['displaced']:4d}")

    print("\nSpace-only pauses, median cost per pause frame:")
    for outcome in ["on its gap", "neighbour phoneme", "displaced"]:
        selected = [p for p in placements if p.has_punctuation is False and p.outcome == outcome]
        if not selected:
            continue
        gap_cost = np.median([p.gap_cost for p in selected])
        competitor_cost = np.median([p.competitor_cost for p in selected])
        competitor_cheaper = np.mean([p.competitor_cost < p.gap_cost for p in selected])
        distance = np.median([p.competitor_distance for p in selected])
        print(f"  {outcome:18} n={len(selected):4d}  on the gap {gap_cost:6.1f}  competitor {competitor_cost:6.1f}  "
              f"competitor cheaper in {competitor_cheaper:4.0%}  competitor {distance:.0f} symbols away")

    displaced = [p for p in placements if p.outcome == "displaced"]
    print("\nDisplaced pauses, competitor symbols:", Counter(p.competitor for p in displaced).most_common(10))
    print("Displaced pauses, words moved:", Counter(p.words_moved for p in displaced).most_common(6))
    print("Displaced pauses, longest 20:")
    for p in sorted(displaced, key=lambda p: -p.milliseconds)[:20]:
        print(f"  {p.audio}  {p.start_seconds:.2f} s  {p.milliseconds:.0f} ms  given to '{p.chosen_symbol}'  {p.words_moved} words away")


def remember_languages(filelist_paths):
    for filelist_path in filelist_paths:
        with open(filelist_path, encoding="utf-8") as f:
            for line in f:
                cells = line.split("|")
                if len(cells) >= 3:
                    LANGUAGE_BY_AUDIO[cells[0].strip()] = cells[2].strip()


def read_transcript(transcript_path: Path, mel_dir: Path):
    """
    Reads a transcript that has no phoneme ids yet, such as a corpus metadata.N.csv being edited, and phonemizes it,
    so the edited commas can be checked before the training filelists are regenerated.
    Lines whose recording has no precomputed mel are skipped and reported.
    """
    from matcha.text.phonemizers import multilingual_phonemizer

    utterances_by_speaker = {}
    without_mel = []
    with open(transcript_path, encoding="utf-8") as f:
        for line in f:
            cells = [cell.strip() for cell in line.rstrip("\n").split("|")]
            is_entry = len(cells) >= 4 and cells[0] != ""
            if not is_entry:
                continue
            relative_path, speaker, language, text = cells[:4]
            LANGUAGE_BY_AUDIO[relative_path] = language
            has_mel = (mel_dir / f"{relative_path}.fine.npy").exists()
            if not has_mel:
                without_mel.append(relative_path)
                continue
            _, phoneme_ids = multilingual_phonemizer(text, language)
            utterance = Utterance(relative_path, int(speaker), text, list(phoneme_ids))
            utterances_by_speaker.setdefault(int(speaker), []).append(utterance)
    if without_mel:
        print(f"skipped {len(without_mel)} lines without a precomputed mel: {' '.join(without_mel[:20])}")
    return utterances_by_speaker


def main():
    parser = argparse.ArgumentParser(description="Where MAS puts each pause, and why.")
    parser.add_argument("-i", "--data-config", required=True)
    parser.add_argument("-c", "--checkpoint", required=True)
    parser.add_argument("-u", "--utterances", nargs="+", default=None, help="Report these recordings in detail")
    parser.add_argument("--files_per_speaker", type=int, default=150, help="0 measures every file")
    parser.add_argument("--speakers", type=int, nargs="+", default=None, help="Only these speakers")
    parser.add_argument("--list_displaced", type=Path, default=None,
                        help="Write every displaced pause here, one line per pause, grouped by speaker")
    parser.add_argument("--list_unpunctuated", type=Path, default=None,
                        help="Write every pause at a word gap without punctuation here, the harmful ones first")
    parser.add_argument("--list_missing_commas", type=Path, default=None,
                        help="Write every pause of at least --min_pause_ms that no punctuation explains, longest first")
    parser.add_argument("--min_pause_ms", type=float, default=110.0)
    parser.add_argument("--proposed_transcript", type=Path, default=None,
                        help="With --transcript: write a copy of it that differs only by the proposed commas")
    parser.add_argument("--silence_db", type=float, default=SILENCE_THRESHOLD_DB,
                        help="A fine frame quieter than this counts as silence; -90 keeps only the visibly flat line")
    parser.add_argument("--transcript", type=Path, default=None,
                        help="Measure this transcript (audio|speaker|language|text, e.g. a corpus metadata.N.csv) "
                             "instead of the training filelist; it is phonemized on the fly")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    data_config = OmegaConf.load(args.data_config)
    train_filelist = Path(data_config.train_filelist_path)
    filelists = [train_filelist, Path(data_config.valid_filelist_path)]
    mel_dir = Path(data_config.mel_dir)
    # The wavs sit next to the mels, so a filelist kept elsewhere (a draft transcript) still finds them.
    wav_dir = mel_dir.parent / "wav"
    loaded = load_encoder(args.checkpoint)

    if args.utterances is not None:
        remember_languages(filelists)
        report_utterances(loaded, args.utterances, filelists, mel_dir, wav_dir)
        return

    # Shorter silences are never listed, so they are not measured either; without a comma list the usual minimum holds.
    min_pause_frames = MIN_PAUSE_FRAMES
    if args.list_missing_commas is not None:
        min_pause_frames = max(1, int(np.ceil(args.min_pause_ms / 1000 / FINE_FRAME_SECONDS)))

    if args.transcript is not None:
        utterances_by_speaker = read_transcript(args.transcript, mel_dir)
    else:
        remember_languages(filelists)
        utterances_by_speaker = read_filelist(train_filelist)

    random.seed(0)
    placements = []
    for speaker, utterances in sorted(utterances_by_speaker.items()):
        is_excluded = args.speakers is not None and speaker not in args.speakers
        if is_excluded:
            continue
        sample = utterances
        if args.files_per_speaker > 0:
            sample = random.sample(utterances, min(args.files_per_speaker, len(utterances)))
        for utterance in sample:
            placements.extend(measure_utterance(loaded, utterance, mel_dir, wav_dir, args.silence_db, min_pause_frames)[0])
    report_corpus(placements, args.files_per_speaker, loaded.epoch)
    if args.list_displaced is not None:
        write_displaced_list(args.list_displaced, placements, loaded.epoch)
    if args.list_unpunctuated is not None:
        write_unpunctuated_list(args.list_unpunctuated, placements, loaded.epoch)
    if args.list_missing_commas is not None:
        write_missing_comma_list(args.list_missing_commas, placements, args.min_pause_ms, args.silence_db, texts_of(utterances_by_speaker))
    if args.proposed_transcript is not None and args.transcript is not None:
        write_proposed_transcript(args.transcript, args.proposed_transcript, placements, args.min_pause_ms)


def texts_of(utterances_by_speaker) -> Dict[str, str]:
    return {u.relative_path: u.text for utterances in utterances_by_speaker.values() for u in utterances}


def write_missing_comma_list(path: Path, placements: List[PausePlacement], min_pause_ms: float, silence_db: float,
                             texts: Dict[str, str]):
    """
    Pauses long enough to be the speaker's comma pause, with no punctuation next to them, for a listener who writes
    a comma wherever the speaker pauses. One line per recording, in recording order, with all its unexplained pauses.

    A pause counts as explained when either the reference position or the alignment search puts it at punctuation:
    the reference can land one word off when neighbouring words sound alike, and the alignment search usually finds
    a written comma, so needing both to miss keeps false alarms down.
    """
    long_pauses = [p for p in placements if p.milliseconds >= min_pause_ms]
    explained = [p for p in long_pauses if p.has_punctuation is True or p.chosen_on_punctuation]
    unexplained = [p for p in long_pauses if p not in explained]
    pauses_by_audio = {}
    for p in unexplained:
        pauses_by_audio.setdefault(p.audio, []).append(p)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Pauses of {min_pause_ms:.0f} ms or more below {silence_db:.0f} dB that no punctuation explains.\n")
        f.write("Times refer to data/corpus-24k/wav. The time is reliable; the words around ⟨⟩ can be one word off, "
                "so listen at the time.\n")
        f.write("Marks: [word] the reference puts the pause inside a word (mostly a missed word gap); "
                "[comma?] a comma sits just before ⟨⟩, so it probably already holds the pause.\n")
        f.write(f"{len(long_pauses)} long pauses, {len(explained)} at punctuation, {len(unexplained)} listed "
                f"in {len(pauses_by_audio)} recordings.\n\n")
        for audio in sorted(pauses_by_audio, key=recording_order):
            pauses = sorted(pauses_by_audio[audio], key=lambda p: p.start_seconds)
            f.write(f"{audio}\t{texts.get(audio, '')}\n")
            for p in pauses:
                mark = ""
                if p.has_punctuation is None:
                    mark = " [word]"
                if has_punctuation_just_before(p.context):
                    mark = " [comma?]"
                f.write(f"\t{p.start_seconds:.2f} s\t{p.milliseconds:4.0f} ms{mark}\t{p.context}\n")
    print(f"\nwrote {len(unexplained)} pauses in {len(pauses_by_audio)} recordings to {path}")


def write_proposed_transcript(transcript_path: Path, out_path: Path, placements: List[PausePlacement],
                              min_pause_ms: float):
    """
    Writes a copy of the transcript that differs only by the proposed commas, so a side-by-side diff jumps from one
    proposal to the next. Every other byte is kept: line order, empty lines, spacing and line endings.

    A comma is proposed after the word before each listed pause, except where a comma probably already holds it
    ([comma?]) or the phonemized text has a different word count than the transcript (numbers expanded by the text
    normalizer), because then the word position cannot be trusted.
    """
    long_pauses = [p for p in placements if p.milliseconds >= min_pause_ms]
    proposals = {}
    unplaceable = []
    for p in long_pauses:
        is_explained = p.has_punctuation is True or p.chosen_on_punctuation
        if is_explained or has_punctuation_just_before(p.context):
            continue
        if not p.words_match_text or p.word_before < 0:
            unplaceable.append(p.audio)
            continue
        proposals.setdefault(p.audio, set()).add(p.word_before)

    changed_lines = 0
    with open(transcript_path, encoding="utf-8", newline="") as source, open(out_path, "w", encoding="utf-8", newline="") as target:
        for line in source:
            cells = line.split("|")
            audio = cells[0].strip()
            has_proposal = len(cells) >= 4 and audio in proposals
            if not has_proposal:
                target.write(line)
                continue
            cells[3] = text_with_commas(cells[3], proposals[audio])
            target.write("|".join(cells))
            changed_lines += 1
    print(f"wrote {out_path}: {changed_lines} lines with proposed commas")
    if unplaceable:
        print(f"no comma proposed (the transcript words could not be matched to the phonemized words): {' '.join(sorted(set(unplaceable)))}")


def text_with_commas(text: str, words_before_pauses) -> str:
    """Puts a comma after each given word (index into the whitespace-separated words), before any closing quote."""
    word_spans = [match.span() for match in re.finditer(r"\S+", text)]
    pieces = []
    previous_end = 0
    for index, (start, end) in enumerate(word_spans):
        word = text[start:end]
        if index in words_before_pauses:
            bare = word.rstrip(CLOSING_QUOTES)
            word = bare + "," + word[len(bare):]
        pieces.append(text[previous_end:start] + word)
        previous_end = end
    pieces.append(text[previous_end:])
    return "".join(pieces)


CLOSING_QUOTES = "\"'”’»)"


def recording_order(audio: str) -> Tuple[int, str]:
    speaker, _, number = audio.partition("/")
    return int(speaker), number


def has_punctuation_just_before(context: str) -> bool:
    words_before = context.split(" ⟨⟩ ")[0].split()
    last_two = words_before[-2:]
    return any(strip_word(w)[-1:] in PUNCTUATION for w in last_two if strip_word(w))


def strip_word(word: str) -> str:
    return word.rstrip("\"'”’»)")


# Pauses inside a word shorter than this are mostly stop closures, so they are left out of the unpunctuated list.
MIN_LISTED_PAUSE_INSIDE_WORD_MS = 150


def write_unpunctuated_list(path: Path, placements: List[PausePlacement], epoch: int):
    """
    Pauses that no punctuation explains, for a listener deciding where a comma belongs. The speaker's typical pause
    at written punctuation is given as a yardstick.
    """
    at_punctuation = [p.milliseconds for p in placements if p.has_punctuation is True]
    unpunctuated = [p for p in placements if p.has_punctuation is False]
    harmful = sorted((p for p in unpunctuated if p.outcome != "on its gap"), key=lambda p: -p.milliseconds)
    on_gap = sorted((p for p in unpunctuated if p.outcome == "on its gap"), key=lambda p: -p.milliseconds)
    inside_word = sorted((p for p in placements if p.has_punctuation is None and p.milliseconds >= MIN_LISTED_PAUSE_INSIDE_WORD_MS),
                         key=lambda p: -p.milliseconds)

    def write_section(f, title, selected):
        f.write(f"\n== {title} ({len(selected)})\n")
        for p in selected:
            f.write(f"{p.audio}\t{p.start_seconds:.2f} s\t{p.milliseconds:4.0f} ms\t{p.outcome}, given to '{p.chosen_symbol}'\t{p.context}\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Pauses at word gaps without punctuation, epoch {epoch}. Times refer to data/corpus-24k/wav.\n")
        f.write("⟨⟩ marks where the pause is according to the reference; each section is sorted longest first.\n")
        if at_punctuation:
            quartiles = np.percentile(at_punctuation, [25, 50, 75])
            f.write(f"Yardstick: pauses at written punctuation, {len(at_punctuation)} pauses, "
                    f"25/50/75th percentile {quartiles[0]:.0f} / {quartiles[1]:.0f} / {quartiles[2]:.0f} ms\n")
        write_section(f, "HARMFUL: the alignment search moved the pause away from its gap", harmful)
        write_section(f, "PLACED ON ITS GAP: harmless for the alignment, a comma only if it is a phrase break", on_gap)
        write_section(f, f"INSIDE A WORD, {MIN_LISTED_PAUSE_INSIDE_WORD_MS} ms or more: a cut, a noise gate or a reference error", inside_word)
    print(f"\nwrote {len(harmful)} harmful, {len(on_gap)} placed on their gap and {len(inside_word)} inside-word pauses to {path}")


def write_displaced_list(path: Path, placements: List[PausePlacement], epoch: int):
    displaced = [p for p in placements if p.outcome == "displaced"]
    displaced.sort(key=lambda p: (p.speaker, p.audio, p.start_seconds))
    likely_real = [p for p in displaced if not p.chosen_on_punctuation]
    likely_reference_error = [p for p in displaced if p.chosen_on_punctuation]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Pauses the alignment search moves away from their word gap, epoch {epoch}.\n")
        f.write("Times refer to data/corpus-24k/wav. ⟨⟩ marks where the pause is according to the reference.\n")
        f.write(f"\n== LIKELY REAL: the pause went to a symbol inside a word or to an unpunctuated gap ({len(likely_real)})\n")
        f.write("audio\ttime\tlength\ttext around the pause\n")
        for p in likely_real:
            f.write(f"{p.audio}\t{p.start_seconds:.2f} s\t{p.milliseconds:.0f} ms\tgiven to '{p.chosen_symbol}'\t{p.context}\n")
        f.write(f"\n== LIKELY FINE: the pause went to a gap with punctuation, the reference is probably one word late "
                f"({len(likely_reference_error)})\n")
        for p in likely_reference_error:
            f.write(f"{p.audio}\t{p.start_seconds:.2f} s\t{p.milliseconds:.0f} ms\t{p.context}\n")
    print(f"\nwrote {len(likely_real)} likely real and {len(likely_reference_error)} likely fine displaced pauses to {path}")


if __name__ == "__main__":
    main()
