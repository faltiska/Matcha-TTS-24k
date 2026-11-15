"""
Aligns a recording with a different text than the one in the filelist, to see whether a change in the text
(a comma, a corrected word) changes where the alignment search (MAS) and the Duration Predictor (DP) put the words.

The alternative text is phonemized with the current phonemizer; the filelist text is phonemized too, and the
script stops if that does not reproduce the filelist's phoneme ids, because then the comparison would not be fair.

Runs on the CPU. Times refer to the matcha wavs (data/corpus-24k/wav).

Usage:
  python -m matcha.utils.align_with_text -i configs/data/corpus-24k-v25.yaml -u 4/1423 \
      --text "In some form or other, we need fun." -c <ckpt> [<ckpt> ...]
"""

import argparse
from pathlib import Path

import torch
from omegaconf import OmegaConf

from matcha.text.phonemizers import multilingual_phonemizer
from matcha.utils.alignment_diagnostics import (
    FINE_FRAME_SECONDS,
    WORD_GAP_LABEL,
    Utterance,
    alignment_scores,
    find_utterance,
    load_encoder,
    load_fine_mel,
    monotonic_alignment_durations,
    rounded_predicted_durations,
    run_encoder,
    word_segments,
)


def language_of(filelist_paths, relative_path: str) -> str:
    """The filelist's language column; the shared Utterance type does not keep it."""
    for filelist_path in filelist_paths:
        with open(filelist_path, encoding="utf-8") as f:
            for line in f:
                if line.startswith(relative_path + "|"):
                    return line.split("|")[2]
    raise KeyError(f"{relative_path} is in none of {filelist_paths}")


def words_text(utterance, durations) -> str:
    parts = []
    for segment in word_segments(utterance, durations):
        if segment.label == WORD_GAP_LABEL:
            continue
        start = segment.start_frame * FINE_FRAME_SECONDS
        end = (segment.start_frame + segment.frames) * FINE_FRAME_SECONDS
        parts.append(f"{segment.label} {start:.2f}-{end:.2f}")
    return " | ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Align a recording with an alternative text.")
    parser.add_argument("-i", "--data-config", required=True)
    parser.add_argument("-u", "--utterance", required=True, help="Relative path as in the filelist, e.g. 4/1423")
    parser.add_argument("--text", required=True, help="The alternative text")
    parser.add_argument("-c", "--checkpoints", required=True, nargs="+")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    data_config = OmegaConf.load(args.data_config)
    filelists = [Path(data_config.train_filelist_path), Path(data_config.valid_filelist_path)]
    original = find_utterance(filelists, args.utterance)
    language = language_of(filelists, args.utterance)

    _, reproduced_ids = multilingual_phonemizer(original.text, language)
    phonemizer_matches_filelist = list(reproduced_ids) == original.phoneme_ids
    if not phonemizer_matches_filelist:
        raise SystemExit(f"The phonemizer no longer reproduces the filelist ids of {args.utterance}; regenerate the filelist first.")
    _, alternative_ids = multilingual_phonemizer(args.text, language)
    alternative = Utterance(original.relative_path, original.speaker, args.text, list(alternative_ids))
    mel = load_fine_mel(Path(data_config.mel_dir), original.relative_path)

    for checkpoint_path in args.checkpoints:
        loaded = load_encoder(checkpoint_path)
        print(f"\n== {checkpoint_path} (epoch {loaded.epoch})")
        for name, utterance in [("filelist text", original), ("alternative", alternative)]:
            encoder_frames, predicted = run_encoder(loaded, utterance)
            mas_durations = monotonic_alignment_durations(alignment_scores(encoder_frames, mel))
            print(f"{name}: '{utterance.text}'")
            print(f"   MAS: {words_text(utterance, mas_durations)}")
            print(f"   DP : {words_text(utterance, rounded_predicted_durations(predicted))}")


if __name__ == "__main__":
    main()
