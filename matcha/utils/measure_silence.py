"""
Measure trailing silence per speaker in a corpus.

Usage:
  python -m matcha.utils.measure_silence -i configs/data/corpus-small-24k.yaml
  
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


def measure_trailing_silence(wav_path: Path, effective_silence_threshold: float, absolute_silence_threshold: float, debug: bool = False) -> Tuple[float, float]:
    """
    Measure trailing silence duration in seconds using RMS in 10ms windows.
    Measures two sections: effective silence (e.g. -60dB) and absolute silence (e.g., -90dB).
    
    Returns:
        Tuple of (effective_silence_duration, absolute_silence_duration) in seconds
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
        
        # Count trailing silence for both thresholds
        is_below_effective_silence_threshold = rms < effective_silence_threshold_amp
        is_below_absolute_silence_threshold = rms < absolute_silence_threshold_amp
        
        # Count effective silence
        windows_below_effective_silence_threshold = 0
        for i in range(len(is_below_effective_silence_threshold) - 1, -1, -1):
            if is_below_effective_silence_threshold[i]:
                windows_below_effective_silence_threshold += 1
            else:
                break
        
        # Count absolute silence
        windows_below_absolute_silence_threshold = 0
        for i in range(len(is_below_absolute_silence_threshold) - 1, -1, -1):
            if is_below_absolute_silence_threshold[i]:
                windows_below_absolute_silence_threshold += 1
            else:
                break
        
        effective_silence_duration = (windows_below_effective_silence_threshold * window_frames) / sr
        absolute_silence_duration = (windows_below_absolute_silence_threshold * window_frames) / sr
        
        if debug:
            print(f"\n{wav_path.name}: sr={sr}, soft={effective_silence_duration*1000:.1f}ms, hard={absolute_silence_duration*1000:.1f}ms")
        
        return effective_silence_duration, absolute_silence_duration
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        return 0.0, 0.0


def main():
    parser = argparse.ArgumentParser(
        description="Measure trailing silence per speaker in a corpus."
    )
    parser.add_argument(
        "-i",
        "--data-config",
        help="Path to data YAML (e.g., configs/data/corpus-small-24k.yaml)",
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
        
        soft, hard = measure_trailing_silence(wav_path, args.effective_silence_threshold, args.absolute_silence_threshold, debug=True)
        print(f"\nSoft silence (-70dB): {soft * 1000:.1f} ms")
        print(f"Hard silence (-90dB): {hard * 1000:.1f} ms")
        return
    
    # Corpus mode
    if not args.data_config:
        print("Error: Either --data-config or --file must be provided")
        return

    cfg_path = Path(args.data_config).resolve()
    cfg = _load_yaml_config(cfg_path)

    train_filelist = _resolve_path(str(cfg["train_filelist_path"]))

    # Print header first
    print(f"\nTrailing Silence Statistics (soft: {args.effective_silence_threshold} dB, hard: {args.absolute_silence_threshold} dB)")
    print("=" * 98)
    print(f"{'Speaker':<10} {'Count':<8} {'Effective Mean':<16} {'Effective Std':<16} {'Absolute Mean':<16} {'Absolute Std':<16}")
    print("-" * 98)
    
    # Collect measurements per speaker
    speaker_soft_silences = defaultdict(list)
    speaker_hard_silences = defaultdict(list)
    speaker_max_soft = {}  # Track file with longest soft silence per speaker
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
            
            soft_duration, hard_duration = measure_trailing_silence(wav_path, args.effective_silence_threshold, args.absolute_silence_threshold, debug=False)
            speaker_soft_silences[speaker_id].append(soft_duration)
            speaker_hard_silences[speaker_id].append(hard_duration)
            
            # Track file with longest soft silence
            if speaker_id not in speaker_max_soft or soft_duration > speaker_max_soft[speaker_id][1]:
                speaker_max_soft[speaker_id] = (str(wav_path), soft_duration)
            
            total += 1
            
            if total % 100 == 0:
                # Only print stats for the current speaker
                soft_ms = np.array(speaker_soft_silences[speaker_id]) * 1000
                hard_ms = np.array(speaker_hard_silences[speaker_id]) * 1000
                soft_mean = np.mean(soft_ms)
                soft_std = np.std(soft_ms)
                hard_mean = np.mean(hard_ms)
                hard_std = np.std(hard_ms)
                count = len(soft_ms)
                
                # Print new line only if speaker changed
                if speaker_id != last_printed_speaker:
                    if last_printed_speaker is not None:
                        print()  # New line for new speaker
                    last_printed_speaker = speaker_id
                
                print(f"\r{speaker_id:<10} {count:<8} {soft_mean:<16.1f} {soft_std:<16.1f} {hard_mean:<16.1f} {hard_std:<16.1f}", end="", flush=True)
    
    print()  # New line after progress
    print("=" * 98)
    print(f"Total files processed: {total}")
    
    # Print files with longest silence per speaker
    print("\nFiles with longest trailing soft silence per speaker:")
    print("-" * 98)
    for speaker_id in sorted(speaker_max_soft.keys()):
        file_path, duration_sec = speaker_max_soft[speaker_id]
        print(f"Speaker {speaker_id}: {duration_sec*1000:.1f}ms - {file_path}")


if __name__ == "__main__":
    main()
