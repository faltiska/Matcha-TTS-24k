"""
Inspect the alignment of one recording under one or more checkpoints.

For each checkpoint it prints, per phoneme group, where the alignment search (MAS) places it on the ground truth
recording and where the Duration Predictor (DP) would place it at inference. MAS durations are the model's own
choice, not ground truth phoneme boundaries. Groups squeezed to one frame per symbol are flagged. It also lists the
silences found directly in the wav, which is what a listener hears as pauses.

It then prints the same comparison per word, which is easier to read against a listening test.

Optional outputs:
  --labels_dir  For the last checkpoint given, Audacity label tracks: the MAS durations (load on top of the corpus
                wav) and the DP durations (load on top of an inference wav made with the same checkpoint).
  --wav_dir     Renders through Vocos: the ground truth mel once, then for every checkpoint the Encoder mel built
                with the MAS durations and the Encoder mel built with the DP durations. File names carry the epoch.
                Comparing the two renders of one checkpoint tells Encoder sound errors apart from DP errors;
                comparing across checkpoints shows when either one changes.

Runs on the CPU, so it can be used while training holds the GPU.

Usage:
  python -m matcha.utils.inspect_alignment -i configs/data/corpus-24k.yaml -u 4/1423 -c <ckpt> [<ckpt> ...] \
      [--labels_dir .] [--wav_dir /tmp]
"""

import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import soundfile as sf
import torch
from omegaconf import OmegaConf

from matcha.utils.alignment_diagnostics import (
    FINE_FRAME_SECONDS,
    SAMPLE_RATE,
    WORD_GAP_LABEL,
    find_utterance,
    load_encoder,
    load_fine_mel,
    monotonic_alignment_durations,
    alignment_scores,
    phoneme_groups,
    rounded_predicted_durations,
    run_encoder,
    write_audacity_labels,
    word_segments,
)
from matcha.utils.model import DEFAULT_DOWNSAMPLER, denormalize, generate_path, get_downsampler

# Silences inside the speech are measured on 10ms windows of the wav.
SILENCE_WINDOW_SECONDS = 0.01
# A window louder than this counts as speech when finding where speech starts and ends.
SPEECH_THRESHOLD_DB = -45.0
# A window quieter than this counts as silence.
SILENCE_THRESHOLD_DB = -50.0

# Groups that are not voiced phonemes never collapse in a meaningful way, so they are not flagged.
NOT_FLAGGED_LABELS = {WORD_GAP_LABEL, "ˈ", "ˌ", "ː", "."}


def inner_silences(wav: np.ndarray) -> Tuple[float, float, List[Tuple[float, float]]]:
    """Returns where speech starts and ends, and every silent stretch between those two points, in seconds."""
    window = int(SILENCE_WINDOW_SECONDS * SAMPLE_RATE)
    window_count = len(wav) // window
    windows = wav[:window_count * window].reshape(window_count, window)
    level_db = 20 * np.log10(np.sqrt((windows ** 2).mean(axis=1)) + 1e-9)

    speech_windows = np.where(level_db > SPEECH_THRESHOLD_DB)[0]
    first_speech, last_speech = speech_windows[0], speech_windows[-1]

    silences = []
    silence_start = None
    for index in range(first_speech, last_speech + 1):
        is_silent = level_db[index] < SILENCE_THRESHOLD_DB
        if is_silent and silence_start is None:
            silence_start = index
        if not is_silent and silence_start is not None:
            silences.append((silence_start * SILENCE_WINDOW_SECONDS, index * SILENCE_WINDOW_SECONDS))
            silence_start = None
    speech_end = (last_speech + 1) * SILENCE_WINDOW_SECONDS
    return first_speech * SILENCE_WINDOW_SECONDS, speech_end, silences


def print_alignment_table(utterance, aligned, predicted):
    print(f"{'phoneme':>7}   {'MAS durations':>24}   {'DP durations':>20}")
    aligned_groups = phoneme_groups(utterance.phoneme_ids, aligned)
    predicted_groups = phoneme_groups(utterance.phoneme_ids, predicted)
    for aligned_group, predicted_group in zip(aligned_groups, predicted_groups):
        symbol_count = aligned_group.last_symbol - aligned_group.first_symbol + 1
        is_squeezed = aligned_group.frames <= symbol_count and aligned_group.label not in NOT_FLAGGED_LABELS
        flag = ""
        if is_squeezed:
            flag = "  <- one frame per symbol"
        aligned_start = aligned_group.start_frame * FINE_FRAME_SECONDS
        aligned_end = (aligned_group.start_frame + aligned_group.frames) * FINE_FRAME_SECONDS
        predicted_start = predicted_group.start_frame * FINE_FRAME_SECONDS
        predicted_end = (predicted_group.start_frame + predicted_group.frames) * FINE_FRAME_SECONDS
        print(
            f"{aligned_group.label:>7}   {aligned_start:6.3f} - {aligned_end:6.3f} s ({aligned_group.frames:3d})"
            f"   {predicted_start:6.3f} - {predicted_end:6.3f} s{flag}"
        )


def print_word_table(utterance, aligned, predicted):
    print(f"\n{'word':>12}   {'MAS durations':>24}   {'DP durations':>24}")
    aligned_words = word_segments(utterance, aligned)
    predicted_words = word_segments(utterance, predicted)
    for aligned_word, predicted_word in zip(aligned_words, predicted_words):
        aligned_start = aligned_word.start_frame * FINE_FRAME_SECONDS
        aligned_end = (aligned_word.start_frame + aligned_word.frames) * FINE_FRAME_SECONDS
        predicted_start = predicted_word.start_frame * FINE_FRAME_SECONDS
        predicted_end = (predicted_word.start_frame + predicted_word.frames) * FINE_FRAME_SECONDS
        print(
            f"{aligned_word.label:>12}   {aligned_start:6.3f} - {aligned_end:6.3f} s ({aligned_word.frames:3d})"
            f"   {predicted_start:6.3f} - {predicted_end:6.3f} s ({predicted_word.frames:3d})"
        )


def write_wav(vocos, out_path: Path, fine_mel: torch.Tensor, loaded):
    statistics = loaded.hyper_parameters["data_statistics"]
    downsample = get_downsampler(loaded.hyper_parameters.get("downsampler", DEFAULT_DOWNSAMPLER))
    with torch.no_grad():
        standard_mel = downsample(fine_mel[None])
        audio = vocos.decode(denormalize(standard_mel, statistics["mel_mean"], statistics["mel_std"]))[0].numpy()
    peak = max(1.0, float(np.abs(audio).max()))
    sf.write(out_path, audio / peak, SAMPLE_RATE)
    print("wrote", out_path)


def encoder_mel_with(encoder_frames: torch.Tensor, durations: np.ndarray) -> torch.Tensor:
    """Repeats each symbol's Encoder frame for its number of frames, as inference does before the Decoder."""
    durations_tensor = torch.from_numpy(durations)[None].float()
    total_frames = int(durations.sum())
    path_mask = torch.ones(1, len(durations), total_frames)
    path = generate_path(durations_tensor, path_mask)[0]
    return encoder_frames @ path


def render_encoder_wavs(vocos, wav_dir: Path, prefix: str, encoder_frames, aligned, predicted, loaded):
    epoch_prefix = f"{prefix}_epoch{loaded.epoch:03d}"
    renders = {
        "encoder_mas_durations": encoder_mel_with(encoder_frames, aligned),
        "encoder_dp_durations": encoder_mel_with(encoder_frames, predicted),
    }
    for name, fine_mel in renders.items():
        write_wav(vocos, wav_dir / f"{epoch_prefix}_{name}.wav", fine_mel, loaded)


def main():
    parser = argparse.ArgumentParser(description="Inspect the alignment of one recording under one or more checkpoints.")
    parser.add_argument("-i", "--data-config", required=True, help="Path to data YAML, e.g. configs/data/corpus-24k.yaml")
    parser.add_argument("-u", "--utterance", required=True, help="Relative path as in the filelist, e.g. 4/1423")
    parser.add_argument("-c", "--checkpoints", required=True, nargs="+")
    parser.add_argument("--labels_dir", type=Path, default=None, help="Write Audacity label tracks here")
    parser.add_argument("--wav_dir", type=Path, default=None, help="Write the Vocos renders here, two per checkpoint plus the ground truth")
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    data_config = OmegaConf.load(args.data_config)
    filelists = [Path(data_config.valid_filelist_path), Path(data_config.train_filelist_path)]
    utterance = find_utterance(filelists, args.utterance)
    mel = load_fine_mel(Path(data_config.mel_dir), utterance.relative_path)
    wav, _ = sf.read(Path(data_config.train_filelist_path).parent / "wav" / f"{utterance.relative_path}.wav", dtype="float32")

    print(f"{utterance.relative_path}  speaker {utterance.speaker}  '{utterance.text}'")
    speech_start, speech_end, silences = inner_silences(wav)
    silence_text = ", ".join(f"{start:.2f}-{end:.2f} s" for start, end in silences) or "none"
    print(f"wav: speech {speech_start:.2f} - {speech_end:.2f} s; silences inside the speech: {silence_text}")

    prefix = utterance.relative_path.replace("/", "_")
    vocos = None
    if args.wav_dir is not None:
        from vocos import Vocos

        args.wav_dir.mkdir(parents=True, exist_ok=True)
        vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval()

    for checkpoint_path in args.checkpoints:
        loaded = load_encoder(checkpoint_path)
        encoder_frames, predicted_raw = run_encoder(loaded, utterance)
        aligned = monotonic_alignment_durations(alignment_scores(encoder_frames, mel))
        predicted = rounded_predicted_durations(predicted_raw)
        print(f"\n== {checkpoint_path} (epoch {loaded.epoch})")
        print_alignment_table(utterance, aligned, predicted)
        print_word_table(utterance, aligned, predicted)
        if vocos is not None:
            render_encoder_wavs(vocos, args.wav_dir, prefix, encoder_frames, aligned, predicted, loaded)

    if args.labels_dir is not None:
        args.labels_dir.mkdir(parents=True, exist_ok=True)
        mas_labels = args.labels_dir / f"{prefix}_mas_labels.txt"
        dp_labels = args.labels_dir / f"{prefix}_dp_labels.txt"
        write_audacity_labels(mas_labels, phoneme_groups(utterance.phoneme_ids, aligned))
        write_audacity_labels(dp_labels, phoneme_groups(utterance.phoneme_ids, predicted))
        print(f"\nwrote {mas_labels} and {dp_labels} (epoch {loaded.epoch})")
    if vocos is not None:
        # The ground truth mel does not depend on the checkpoint; only the mel statistics are taken from it.
        write_wav(vocos, args.wav_dir / f"{prefix}_ground_truth.wav", mel, loaded)


if __name__ == "__main__":
    main()
