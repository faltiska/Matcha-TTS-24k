"""
Add or trim trailing silence to wav files to ensure consistent silence across speakers.

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

This script normalizes trailing silence by locating the end of speech content (using
10ms RMS windows anchored from sample 0) and rebuilding each file as:

    speech_content + exactly target_silence_samples zeros

The skip check compares integer sample counts rather than floating-point silence
measurements, so the script is fully idempotent: running it twice produces identical
output.

Usage:
  # Normalize silence in corpus
  python -m matcha.utils.normalize_silence -i configs/data/corpus-24k.yaml --target_silence 0.8

  # Process single file
  python -m matcha.utils.normalize_silence -f input.wav --target_silence 0.8 -o output.wav
"""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Tuple

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


def _find_content_end(audio_mono: torch.Tensor, sr: int, threshold_db: float) -> int:
    """
    Return the sample index just past the last non-silent window.

    Windows of 10ms are always anchored from sample 0, so the result is
    independent of file length. This makes it safe to call repeatedly on
    the same file: the content boundary never drifts.

    A trailing partial window that has been zero-padded for computation is
    treated as silence, which is correct — we never want to preserve padding.
    """
    window_frames = int(0.01 * sr)  # 10ms
    threshold_amp = 10 ** (threshold_db / 20.0)

    audio_squared = audio_mono ** 2
    pad_size = window_frames - (len(audio_squared) % window_frames)
    if pad_size < window_frames:
        audio_squared = torch.nn.functional.pad(audio_squared, (0, pad_size))

    rms = torch.sqrt(audio_squared.reshape(-1, window_frames).mean(dim=1))

    last_active = -1
    for i in range(len(rms) - 1, -1, -1):
        if rms[i] >= threshold_amp:
            last_active = i
            break

    # Clamp to actual audio length so padding never leaks into the output
    return min((last_active + 1) * window_frames, len(audio_mono))


def measure_trailing_silence(audio: torch.Tensor, sr: int, threshold_db: float) -> float:
    """Measure trailing silence duration in seconds (used for reporting only)."""
    content_end = _find_content_end(audio, sr, threshold_db)
    return (len(audio) - content_end) / sr


def normalize_trailing_silence(
        wav_path: Path,
        output_path: Path,
        target_silence_sec: float,
        threshold_db: float = -60.0,
) -> tuple[bool, float, float]:
    """
    Normalize trailing silence of a wav file to exactly the target duration.

    The output is always:  speech_content + int(target_silence_sec * sr) zeros

    The skip check is a simple integer sample-count comparison, so running
    this function twice on the same file is guaranteed to be a no-op on the
    second call regardless of sample rate or target duration.

    Returns:
        (changed, current_silence_sec, silence_delta_sec)
        silence_delta_sec is positive when silence was added, negative when trimmed.
    """
    audio, sr = ta.load(str(wav_path))
    audio_mono = audio[0] if audio.shape[0] > 1 else audio.squeeze(0)

    content_end = _find_content_end(audio_mono, sr, threshold_db)
    target_silence_samples = int(target_silence_sec * sr)
    desired_length = content_end + target_silence_samples

    # Integer comparison: avoids all floating-point window-boundary ambiguity
    if audio.shape[1] == desired_length:
        current_silence = (len(audio_mono) - content_end) / sr
        return False, current_silence, 0.0

    current_silence = (len(audio_mono) - content_end) / sr
    silence_delta = target_silence_sec - current_silence

    silence = torch.zeros((audio.shape[0], target_silence_samples), dtype=audio.dtype)
    normalized = torch.cat([audio[:, :content_end], silence], dim=1)

    ta.save(str(output_path), normalized, sr)
    return True, current_silence, silence_delta


def main():
    parser = argparse.ArgumentParser(
        description="Normalize trailing silence in wav files to a fixed target duration."
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
        "--target_silence",
        type=float,
        required=True,
        help="Target trailing silence duration in seconds (e.g., 0.8)",
    )
    parser.add_argument(
        "--threshold_db",
        type=float,
        default=-60.0,
        help="dB threshold for silence detection (default: -60.0)",
    )
    args = parser.parse_args()

    # Single file mode
    if args.file:
        wav_path = Path(args.file)
        if not wav_path.exists():
            print(f"File not found: {wav_path}")
            return

        output_path = Path(args.output) if args.output else wav_path.parent / f"{wav_path.stem}_silence{wav_path.suffix}"

        changed, current_silence, delta = normalize_trailing_silence(wav_path, output_path, args.target_silence,args.threshold_db)
        if changed:
            action = "Added" if delta > 0 else "Trimmed"
            print(f"{action} {abs(delta) * 1000:.1f}ms silence → {output_path}")
        else:
            print(f"No change needed for {wav_path} (current: {current_silence * 1000:.1f}ms)")
        return

    # Corpus mode
    if not args.data_config:
        print("Error: Either --data-config or --file must be provided")
        return

    cfg_path = Path(args.data_config).resolve()
    cfg = _load_yaml_config(cfg_path)

    train_filelist = _resolve_path(str(cfg["train_filelist_path"]))
    valid_filelist_path = cfg.get("valid_filelist_path", "")
    valid_filelist = _resolve_path(str(valid_filelist_path)) if valid_filelist_path else None

    total = 0
    added = 0
    trimmed = 0

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

            changed, _, delta = normalize_trailing_silence(wav_path, wav_path, args.target_silence, args.threshold_db)
            if changed:
                if delta > 0:
                    added += 1
                else:
                    trimmed += 1

            if total % 100 == 0:
                print(f"Processed {total} files — added: {added}, trimmed: {trimmed}...", end="\r", flush=True,)

    print(f"\nDone. {total} files processed: {added} padded, {trimmed} trimmed, {total - added - trimmed} unchanged.")


if __name__ == "__main__":
    main()
