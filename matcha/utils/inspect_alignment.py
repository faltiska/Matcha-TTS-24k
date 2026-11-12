"""
Inspect the alignment of one recording under one or more checkpoints.

For each checkpoint it prints, per phoneme group, where the alignment places it on the ground truth recording
and where the Duration Predictor would place it at inference. Groups squeezed to one frame per symbol are
flagged. It also lists the silences found directly in the wav, which is what a listener hears as pauses.

Optional outputs, for the last checkpoint given:
  --labels_dir  Audacity label tracks: the ground truth alignment (load on top of the corpus wav) and the
                predicted durations (load on top of an inference wav made with the same checkpoint).
  --wav_dir     Three renders through Vocos: the ground truth mel, the Encoder mel built with the ground truth
                alignment, and the Encoder mel built with the predicted durations. Comparing the last two tells
                Encoder sound errors apart from Duration Predictor errors.

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
    print(f"{'phoneme':>7}   {'ground truth alignment':>24}   {'predicted durations':>20}")
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


def render_wavs(wav_dir: Path, utterance, encoder_frames, aligned, predicted, mel, loaded):
    from vocos import Vocos

    statistics = loaded.hyper_parameters["data_statistics"]
    downsample = get_downsampler(loaded.hyper_parameters.get("downsampler", DEFAULT_DOWNSAMPLER))
    vocos = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval()

    def encoder_mel_with(durations):
        durations_tensor = torch.from_numpy(durations)[None].float()
        total_frames = int(durations.sum())
        path_mask = torch.ones(1, len(durations), total_frames)
        path = generate_path(durations_tensor, path_mask)[0]
        return encoder_frames @ path

    renders = {
        "ground_truth": mel,
        "encoder_ground_truth_alignment": encoder_mel_with(aligned),
        "encoder_predicted_durations": encoder_mel_with(predicted),
    }
    prefix = utterance.relative_path.replace("/", "_")
    with torch.no_grad():
        for name, fine_mel in renders.items():
            standard_mel = downsample(fine_mel[None])
            audio = vocos.decode(denormalize(standard_mel, statistics["mel_mean"], statistics["mel_std"]))[0].numpy()
            peak = max(1.0, float(np.abs(audio).max()))
            out_path = wav_dir / f"{prefix}_{name}.wav"
            sf.write(out_path, audio / peak, SAMPLE_RATE)
            print("wrote", out_path)


def main():
    parser = argparse.ArgumentParser(description="Inspect the alignment of one recording under one or more checkpoints.")
    parser.add_argument("-i", "--data-config", required=True, help="Path to data YAML, e.g. configs/data/corpus-24k.yaml")
    parser.add_argument("-u", "--utterance", required=True, help="Relative path as in the filelist, e.g. 4/1423")
    parser.add_argument("-c", "--checkpoints", required=True, nargs="+")
    parser.add_argument("--labels_dir", type=Path, default=None, help="Write Audacity label tracks here")
    parser.add_argument("--wav_dir", type=Path, default=None, help="Write the three Vocos renders here")
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

    for checkpoint_path in args.checkpoints:
        loaded = load_encoder(checkpoint_path)
        encoder_frames, predicted_raw = run_encoder(loaded, utterance)
        aligned = monotonic_alignment_durations(alignment_scores(encoder_frames, mel))
        predicted = rounded_predicted_durations(predicted_raw)
        print(f"\n== {checkpoint_path} (epoch {loaded.epoch})")
        print_alignment_table(utterance, aligned, predicted)

    prefix = utterance.relative_path.replace("/", "_")
    if args.labels_dir is not None:
        args.labels_dir.mkdir(parents=True, exist_ok=True)
        ground_truth_labels = args.labels_dir / f"{prefix}_ground_truth_labels.txt"
        predicted_labels = args.labels_dir / f"{prefix}_predicted_labels.txt"
        write_audacity_labels(ground_truth_labels, phoneme_groups(utterance.phoneme_ids, aligned))
        write_audacity_labels(predicted_labels, phoneme_groups(utterance.phoneme_ids, predicted))
        print(f"\nwrote {ground_truth_labels} and {predicted_labels} (epoch {loaded.epoch})")
    if args.wav_dir is not None:
        args.wav_dir.mkdir(parents=True, exist_ok=True)
        render_wavs(args.wav_dir, utterance, encoder_frames, aligned, predicted, mel, loaded)


if __name__ == "__main__":
    main()
