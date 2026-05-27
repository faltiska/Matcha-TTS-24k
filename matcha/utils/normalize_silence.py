"""
Add or trim leading and trailing silence in wav files to ensure consistent silence
across speakers, at both ends of every utterance.

Background:
Analysis showed that different speakers in the corpus had significantly different amounts
of trailing silence. Measurement results at -60dB threshold:

  Speaker    Count    Effective Mean (ms)    Effective Std (ms)
  0          4300     361.1                  79.5
  1          4292     742.1                  16.4
  2          1306     807.6                  18.5
  3          1157     347.5                  11.7
  4          1297     13.3                   7.4
  5          852      55.9                   23.3
  6          785      322.8                  18.6
  7          1318     873.0                  11.9
  8          2508     852.5                  21.4
  9          2487     895.0                  19.4

Speakers with more trailing silence (e.g., speaker 9 with 895ms) appeared to be learned
better by the model than those with less (e.g., speaker 0 with 361ms). This suggests the
model was using silence duration as a spurious feature for speaker identification rather
than learning from actual voice characteristics.

The same logic applies to leading silence. We normalize both ends so the model sees a
consistent silence prefix and suffix on every utterance, which also gives the encoder
multiple silence symbols to anchor to at the beginning and end of every utterance.

This script rebuilds each file as:

    target_leading_silence_samples zeros + speech_content + target_trailing_silence_samples zeros

The skip check compares integer sample counts rather than floating-point silence
measurements, so the script is fully idempotent: running it twice produces identical output.

Idempotency requires the targets to be whole multiples of the 10ms window size, otherwise
the window grid would shift relative to the speech content between runs and the detected
content boundary would drift. The script asserts this on startup.

Usage:
  # Normalize both ends in the corpus
  python -m matcha.utils.normalize_silence -i configs/data/corpus-24k.yaml \\
      --target_leading_silence 0.2 --target_trailing_silence 0.8

  # Normalize only the trailing end
  python -m matcha.utils.normalize_silence -i configs/data/corpus-24k.yaml \\
      --target_trailing_silence 0.8

  # Process single file
  python -m matcha.utils.normalize_silence -f input.wav \\
      --target_leading_silence 0.2 --target_trailing_silence 0.8 -o output.wav
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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


def _find_content_bounds(audio_mono: torch.Tensor, sr: int, threshold_db: float) -> Tuple[int, int]:
    """
    Return (content_start_sample, content_end_sample).

    Both bounds are aligned to 10ms window boundaries (anchored at sample 0):
      - content_start = first sample of the first window with RMS at or above threshold
                        (forward scan)
      - content_end   = sample just past the last window with RMS at or above threshold
                        (backward scan)

    Anchoring windows at sample 0 means the same windows always cover the same speech
    content, so the boundary detection is idempotent: as long as we add or remove a
    whole number of windows of zero samples at either end, the next run produces
    identical bounds.

    A trailing partial window that has been zero-padded for computation is treated as
    silence — we never want to preserve padding.

    For a fully silent file, returns (0, 0); the body slice is empty.
    """
    window_frames = int(0.01 * sr)  # 10ms
    threshold_amp = 10 ** (threshold_db / 20.0)

    audio_squared = audio_mono ** 2
    pad_size = window_frames - (len(audio_squared) % window_frames)
    if pad_size < window_frames:
        audio_squared = torch.nn.functional.pad(audio_squared, (0, pad_size))

    rms = torch.sqrt(audio_squared.reshape(-1, window_frames).mean(dim=1))
    is_active = rms >= threshold_amp

    first_active = -1
    for i in range(len(is_active)):
        if is_active[i]:
            first_active = i
            break

    last_active = -1
    for i in range(len(is_active) - 1, -1, -1):
        if is_active[i]:
            last_active = i
            break

    if first_active == -1:
        # Fully silent file: empty body.
        return 0, 0

    content_start = first_active * window_frames
    # Clamp to actual audio length so padding never leaks into the output
    content_end = min((last_active + 1) * window_frames, len(audio_mono))
    return content_start, content_end


def _samples_for_target(target_silence_sec: Optional[float], sr: int, label: str) -> Optional[int]:
    """
    Convert a target silence duration in seconds to integer samples and validate that it is
    a whole multiple of the 10ms window size. Returns None if the target is None.
    """
    if target_silence_sec is None:
        return None

    window_frames = int(0.01 * sr)
    target_samples = int(round(target_silence_sec * sr))
    if target_samples % window_frames != 0:
        raise ValueError(
            f"--target_{label}_silence must be a whole multiple of 10ms "
            f"(got {target_silence_sec * 1000:.3f} ms at sr={sr})"
        )
    return target_samples


def normalize_silence(
        wav_path: Path,
        output_path: Path,
        target_leading_silence_sec: Optional[float],
        target_trailing_silence_sec: Optional[float],
        threshold_db: float = -60.0,
) -> Tuple[bool, float, float, float, float]:
    """
    Normalize leading and/or trailing silence of a wav file to exactly the target durations.

    Either or both targets may be None. A None target leaves that end untouched.

    The skip check is a simple integer sample-count comparison on each end being normalized,
    so running this function twice on the same file is guaranteed to be a no-op on the
    second call regardless of sample rate or target duration (provided the targets are
    whole multiples of 10ms; this is asserted upstream).

    Returns:
        (changed, current_leading_silence_sec, current_trailing_silence_sec,
         leading_delta_sec, trailing_delta_sec)
        Deltas are positive when silence was added, negative when trimmed,
        and 0.0 when that end was not normalized.
    """
    audio, sr = ta.load(str(wav_path))
    audio_mono = audio[0] if audio.shape[0] > 1 else audio.squeeze(0)

    target_leading_samples = _samples_for_target(target_leading_silence_sec, sr, "leading")
    target_trailing_samples = _samples_for_target(target_trailing_silence_sec, sr, "trailing")

    content_start, content_end = _find_content_bounds(audio_mono, sr, threshold_db)
    current_leading_samples = content_start
    current_trailing_samples = audio.shape[1] - content_end
    current_leading_silence = current_leading_samples / sr
    current_trailing_silence = current_trailing_samples / sr

    leading_already_correct = (
        target_leading_samples is None or current_leading_samples == target_leading_samples
    )
    trailing_already_correct = (
        target_trailing_samples is None or current_trailing_samples == target_trailing_samples
    )
    if leading_already_correct and trailing_already_correct:
        return False, current_leading_silence, current_trailing_silence, 0.0, 0.0

    # Build the new audio as: [leading silence] + [body] + [trailing silence]
    if target_leading_samples is not None:
        leading_part = torch.zeros((audio.shape[0], target_leading_samples), dtype=audio.dtype)
        leading_delta = (target_leading_samples - current_leading_samples) / sr
    else:
        leading_part = audio[:, :content_start]
        leading_delta = 0.0

    if target_trailing_samples is not None:
        trailing_part = torch.zeros((audio.shape[0], target_trailing_samples), dtype=audio.dtype)
        trailing_delta = (target_trailing_samples - current_trailing_samples) / sr
    else:
        trailing_part = audio[:, content_end:]
        trailing_delta = 0.0

    body = audio[:, content_start:content_end]
    normalized = torch.cat([leading_part, body, trailing_part], dim=1)

    ta.save(str(output_path), normalized, sr)
    return True, current_leading_silence, current_trailing_silence, leading_delta, trailing_delta


def _format_delta_message(delta_sec: float, label: str) -> Optional[str]:
    if delta_sec == 0.0:
        return None
    if delta_sec > 0:
        action = "Added"
    else:
        action = "Trimmed"
    return f"{action} {abs(delta_sec) * 1000:.1f}ms {label} silence"


def main():
    parser = argparse.ArgumentParser(
        description="Normalize leading and/or trailing silence in wav files to fixed target durations."
    )
    parser.add_argument(
        "-i", "--data-config",
        help="Path to data YAML (e.g., configs/data/corpus-24k.yaml)",
    )
    parser.add_argument(
        "-f", "--file",
        help="Path to a single wav file to process",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path for single file mode",
    )
    parser.add_argument(
        "--target_leading_silence",
        type=float,
        default=None,
        help="Target leading silence duration in seconds (e.g., 0.2). Must be a multiple of 10ms.",
    )
    parser.add_argument(
        "--target_trailing_silence",
        type=float,
        default=None,
        help="Target trailing silence duration in seconds (e.g., 0.8). Must be a multiple of 10ms.",
    )
    parser.add_argument(
        "--threshold_db",
        type=float,
        default=-60.0,
        help="dB threshold for silence detection (default: -60.0)",
    )
    args = parser.parse_args()

    no_targets_given = args.target_leading_silence is None and args.target_trailing_silence is None
    if no_targets_given:
        print("Error: at least one of --target_leading_silence or --target_trailing_silence must be provided")
        return

    # Single file mode
    if args.file:
        wav_path = Path(args.file)
        if not wav_path.exists():
            print(f"File not found: {wav_path}")
            return

        if args.output:
            output_path = Path(args.output)
        else:
            output_path = wav_path.parent / f"{wav_path.stem}_silence{wav_path.suffix}"

        changed, current_lead, current_tail, lead_delta, tail_delta = normalize_silence(
            wav_path,
            output_path,
            args.target_leading_silence,
            args.target_trailing_silence,
            args.threshold_db,
        )
        if changed:
            messages = []
            leading_message = _format_delta_message(lead_delta, "leading")
            if leading_message is not None:
                messages.append(leading_message)
            trailing_message = _format_delta_message(tail_delta, "trailing")
            if trailing_message is not None:
                messages.append(trailing_message)
            print(", ".join(messages) + f" → {output_path}")
        else:
            print(
                f"No change needed for {wav_path} "
                f"(current leading: {current_lead * 1000:.1f}ms, trailing: {current_tail * 1000:.1f}ms)"
            )
        return

    # Corpus mode
    if not args.data_config:
        print("Error: Either --data-config or --file must be provided")
        return

    cfg_path = Path(args.data_config).resolve()
    cfg = _load_yaml_config(cfg_path)

    train_filelist = _resolve_path(str(cfg["train_filelist_path"]))
    valid_filelist_path = cfg.get("valid_filelist_path", "")
    if valid_filelist_path:
        valid_filelist = _resolve_path(str(valid_filelist_path))
    else:
        valid_filelist = None

    total = 0
    leading_added = 0
    leading_trimmed = 0
    trailing_added = 0
    trailing_trimmed = 0

    filelists = [train_filelist]
    if valid_filelist and valid_filelist.exists() and valid_filelist.is_file():
        filelists.append(valid_filelist)

    for fl in filelists:
        if not fl.exists():
            continue

        entries = parse_filelist(fl)
        for parts in entries:
            if not parts or len(parts) < 2:
                continue

            rel_base = parts[0]
            wav_path = (fl.parent / "wav" / (rel_base + ".wav")).resolve()

            if not wav_path.exists():
                print(f"Warning: {wav_path} not found")
                continue

            total += 1

            _, _, _, lead_delta, tail_delta = normalize_silence(
                wav_path,
                wav_path,
                args.target_leading_silence,
                args.target_trailing_silence,
                args.threshold_db,
            )
            if lead_delta > 0:
                leading_added += 1
            elif lead_delta < 0:
                leading_trimmed += 1
            if tail_delta > 0:
                trailing_added += 1
            elif tail_delta < 0:
                trailing_trimmed += 1

            if total % 100 == 0:
                print(
                    f"Processed {total} files — "
                    f"leading: +{leading_added}/-{leading_trimmed}   "
                    f"trailing: +{trailing_added}/-{trailing_trimmed}",
                    end="\r",
                    flush=True,
                )

    print(
        f"\nDone. {total} files processed. "
        f"Leading: {leading_added} padded, {leading_trimmed} trimmed. "
        f"Trailing: {trailing_added} padded, {trailing_trimmed} trimmed."
    )


if __name__ == "__main__":
    main()
