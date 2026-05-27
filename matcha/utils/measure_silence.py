"""
Measure leading and trailing silence per speaker in a corpus.

Leading silence is found by scanning forward from sample 0; trailing silence by scanning
backward from the end. Both scans share the same window grid (anchored at sample 0) so
they compose cleanly and are independent of file length.

Usage:
  python -m matcha.utils.measure_silence -i configs/data/corpus-24k.yaml

"""

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
import torchaudio as ta


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    try:
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(str(path))
        return dict(OmegaConf.to_container(cfg, resolve=True))
    except Exception:
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


def _resolve_path(maybe_path: str) -> Path:
    p = Path(maybe_path)
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def parse_filelist(filelist_path: Path, split_char: str = "|") -> List[Tuple[str, ...]]:
    with open(filelist_path, encoding="utf-8") as f:
        return [tuple(line.strip().split(split_char)) for line in f if line.strip()]


def _count_leading_silent_windows(is_below_threshold: torch.Tensor) -> int:
    count = 0
    for i in range(len(is_below_threshold)):
        if is_below_threshold[i]:
            count += 1
        else:
            break
    return count


def _count_trailing_silent_windows(is_below_threshold: torch.Tensor) -> int:
    count = 0
    for i in range(len(is_below_threshold) - 1, -1, -1):
        if is_below_threshold[i]:
            count += 1
        else:
            break
    return count


def measure_silence(
        wav_path: Path,
        effective_silence_threshold: float,
        absolute_silence_threshold: float,
        debug: bool = False,
) -> Tuple[float, float, float, float]:
    """
    Measure leading and trailing silence durations in seconds using RMS in 10ms windows.
    Each end is measured at two thresholds: effective silence (e.g. -60dB) and
    absolute silence (e.g., -90dB).

    Returns:
        Tuple of (leading_effective_duration, leading_absolute_duration,
                  trailing_effective_duration, trailing_absolute_duration) in seconds.
    """
    try:
        audio, sr = ta.load(str(wav_path))

        # Convert to mono
        if audio.shape[0] > 1:
            audio = audio[0]
        else:
            audio = audio.squeeze(0)

        effective_silence_threshold_amp = 10 ** (effective_silence_threshold / 20.0)
        absolute_silence_threshold_amp = 10 ** (absolute_silence_threshold / 20.0)

        # Calculate RMS in 10ms windows
        window_frames = int(0.01 * sr)  # 10ms
        audio_squared = audio ** 2

        # Pad to make it divisible by window size
        pad_size = window_frames - (len(audio) % window_frames)
        if pad_size < window_frames:
            audio_squared = torch.nn.functional.pad(audio_squared, (0, pad_size))

        # Reshape and calculate RMS per window
        audio_squared = audio_squared.reshape(-1, window_frames)
        rms = torch.sqrt(audio_squared.mean(dim=1))

        is_below_effective_threshold = rms < effective_silence_threshold_amp
        is_below_absolute_threshold = rms < absolute_silence_threshold_amp

        # Leading silence: scan forward from index 0
        leading_effective_windows = _count_leading_silent_windows(is_below_effective_threshold)
        leading_absolute_windows = _count_leading_silent_windows(is_below_absolute_threshold)

        # Trailing silence: scan backward from the last index
        trailing_effective_windows = _count_trailing_silent_windows(is_below_effective_threshold)
        trailing_absolute_windows = _count_trailing_silent_windows(is_below_absolute_threshold)

        leading_effective_duration = (leading_effective_windows * window_frames) / sr
        leading_absolute_duration = (leading_absolute_windows * window_frames) / sr
        trailing_effective_duration = (trailing_effective_windows * window_frames) / sr
        trailing_absolute_duration = (trailing_absolute_windows * window_frames) / sr

        if debug:
            print(f"\n{wav_path.name}: sr={sr}")
            print(f"  Leading:  effective={leading_effective_duration * 1000:.1f}ms, absolute={leading_absolute_duration * 1000:.1f}ms")
            print(f"  Trailing: effective={trailing_effective_duration * 1000:.1f}ms, absolute={trailing_absolute_duration * 1000:.1f}ms")

        return (
            leading_effective_duration,
            leading_absolute_duration,
            trailing_effective_duration,
            trailing_absolute_duration,
        )
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return 0.0, 0.0, 0.0, 0.0


def _print_silence_table(
        title: str,
        speaker_effective_silences: Dict[str, List[float]],
        speaker_absolute_silences: Dict[str, List[float]],
        effective_threshold_db: float,
        absolute_threshold_db: float,
) -> None:
    print(f"\n{title} (effective: {effective_threshold_db} dB, absolute: {absolute_threshold_db} dB)")
    print("=" * 98)
    print(f"{'Speaker':<10} {'Count':<8} {'Effective Mean':<16} {'Effective Std':<16} {'Absolute Mean':<16} {'Absolute Std':<16}")
    print("-" * 98)
    for speaker_id in sorted(speaker_effective_silences.keys()):
        effective_ms = np.array(speaker_effective_silences[speaker_id]) * 1000
        absolute_ms = np.array(speaker_absolute_silences[speaker_id]) * 1000
        count = len(effective_ms)
        print(f"{speaker_id:<10} {count:<8} {np.mean(effective_ms):<16.1f} {np.std(effective_ms):<16.1f} {np.mean(absolute_ms):<16.1f} {np.std(absolute_ms):<16.1f}")
    print("=" * 98)


def _print_longest_files(title: str, speaker_to_longest: Dict[str, Tuple[str, float]]) -> None:
    print(f"\n{title}:")
    print("-" * 98)
    for speaker_id in sorted(speaker_to_longest.keys()):
        file_path, duration_sec = speaker_to_longest[speaker_id]
        print(f"Speaker {speaker_id}: {duration_sec * 1000:.1f}ms - {file_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Measure leading and trailing silence per speaker in a corpus."
    )
    parser.add_argument(
        "-i",
        "--data-config",
        help="Path to data YAML (e.g., configs/data/corpus-24k.yaml)",
    )
    parser.add_argument(
        "-f",
        "--file",
        help="Path to a single wav file to measure",
    )
    parser.add_argument(
        "--effective_silence_threshold",
        type=float,
        default=-60.0,
        help="dB threshold for effective silence (default: -60.0)",
    )
    parser.add_argument(
        "--absolute_silence_threshold",
        type=float,
        default=-90.0,
        help="dB threshold for absolute silence (default: -90.0)",
    )
    args = parser.parse_args()

    # Single file mode
    if args.file:
        wav_path = Path(args.file)
        if not wav_path.exists():
            print(f"File not found: {wav_path}")
            return

        lead_eff, lead_abs, tail_eff, tail_abs = measure_silence(
            wav_path,
            args.effective_silence_threshold,
            args.absolute_silence_threshold,
            debug=True,
        )
        print(f"\nLeading  effective ({args.effective_silence_threshold} dB): {lead_eff * 1000:.1f} ms")
        print(f"Leading  absolute  ({args.absolute_silence_threshold} dB): {lead_abs * 1000:.1f} ms")
        print(f"Trailing effective ({args.effective_silence_threshold} dB): {tail_eff * 1000:.1f} ms")
        print(f"Trailing absolute  ({args.absolute_silence_threshold} dB): {tail_abs * 1000:.1f} ms")
        return

    # Corpus mode
    if not args.data_config:
        print("Error: Either --data-config or --file must be provided")
        return

    cfg_path = Path(args.data_config).resolve()
    cfg = _load_yaml_config(cfg_path)

    train_filelist = _resolve_path(str(cfg["train_filelist_path"]))

    # Collect measurements per speaker
    speaker_leading_effective = defaultdict(list)
    speaker_leading_absolute = defaultdict(list)
    speaker_trailing_effective = defaultdict(list)
    speaker_trailing_absolute = defaultdict(list)
    speaker_longest_leading = {}  # Track file with longest leading effective silence per speaker
    speaker_longest_trailing = {}  # Track file with longest trailing effective silence per speaker
    last_printed_speaker = None

    total = 0
    for fl in [train_filelist]:
        if not fl.exists():
            raise FileNotFoundError(f"Filelist not found: {fl}")

        entries = parse_filelist(fl)
        for parts in entries:
            if not parts or len(parts) < 2:
                continue

            rel_base = parts[0]
            speaker_id = parts[1]
            wav_path = (fl.parent / "wav" / (rel_base + ".wav")).resolve()

            if not wav_path.exists():
                print(f"Warning: {wav_path} not found")
                continue

            lead_eff, lead_abs, tail_eff, tail_abs = measure_silence(
                wav_path,
                args.effective_silence_threshold,
                args.absolute_silence_threshold,
                debug=False,
            )
            speaker_leading_effective[speaker_id].append(lead_eff)
            speaker_leading_absolute[speaker_id].append(lead_abs)
            speaker_trailing_effective[speaker_id].append(tail_eff)
            speaker_trailing_absolute[speaker_id].append(tail_abs)

            # Track files with the longest effective silence at each end
            longest_leading_so_far = speaker_longest_leading.get(speaker_id)
            if longest_leading_so_far is None or lead_eff > longest_leading_so_far[1]:
                speaker_longest_leading[speaker_id] = (str(wav_path), lead_eff)

            longest_trailing_so_far = speaker_longest_trailing.get(speaker_id)
            if longest_trailing_so_far is None or tail_eff > longest_trailing_so_far[1]:
                speaker_longest_trailing[speaker_id] = (str(wav_path), tail_eff)

            total += 1

            if total % 100 == 0:
                count = len(speaker_leading_effective[speaker_id])
                lead_mean_ms = np.mean(np.array(speaker_leading_effective[speaker_id]) * 1000)
                tail_mean_ms = np.mean(np.array(speaker_trailing_effective[speaker_id]) * 1000)

                # Print new line only if speaker changed
                if speaker_id != last_printed_speaker:
                    if last_printed_speaker is not None:
                        print()  # New line for new speaker
                    last_printed_speaker = speaker_id

                print(
                    f"\rspeaker {speaker_id:<6} {count:<6} files   "
                    f"lead eff mean: {lead_mean_ms:>6.1f} ms   "
                    f"tail eff mean: {tail_mean_ms:>6.1f} ms",
                    end="",
                    flush=True,
                )

    print()  # New line after progress
    print(f"Total files processed: {total}")

    _print_silence_table(
        "Leading Silence Statistics",
        speaker_leading_effective,
        speaker_leading_absolute,
        args.effective_silence_threshold,
        args.absolute_silence_threshold,
    )
    _print_silence_table(
        "Trailing Silence Statistics",
        speaker_trailing_effective,
        speaker_trailing_absolute,
        args.effective_silence_threshold,
        args.absolute_silence_threshold,
    )

    _print_longest_files(
        "Files with longest leading effective silence per speaker",
        speaker_longest_leading,
    )
    _print_longest_files(
        "Files with longest trailing effective silence per speaker",
        speaker_longest_trailing,
    )


if __name__ == "__main__":
    main()
