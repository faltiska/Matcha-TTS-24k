"""
Precompute and cache normalized mel-spectrograms and F0 for a given dataset YAML config.

This script reads all required parameters (filelists, mel params, F0 params, normalization stats,
and output directories) from a Hydra-style data config YAML like configs/data/corpus-small.yaml.

It then saves:
  - Normalized mel spectrograms as .npy files (shape: (n_mels, T)) into mel_dir
  - F0 arrays as .npy files (shape: (1, T)) into f0_dir
  - metadata.json with the parameters used

Usage:
  python -m matcha.utils.precompute_corpus -i configs/data/corpus-small.yaml
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import torch
import torchaudio as ta

from matcha.utils.model import normalize

# NEW: Import mel extractor factory
from matcha.mel.extractors import get_mel_extractor


def _load_yaml_config(path: Path) -> Dict[str, Any]:
    """
    Load a Hydra-style YAML config.
    Tries OmegaConf first (if available) to resolve simple interpolations,
    falls back to PyYAML safe_load otherwise.
    """
    try:
        from omegaconf import OmegaConf  # type: ignore
        cfg = OmegaConf.load(str(path))
        return dict(OmegaConf.to_container(cfg, resolve=True))  # type: ignore
    except Exception:
        # Fallback: PyYAML (no interpolation resolution)
        import yaml  # type: ignore

        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)


def _resolve_path(maybe_path: str) -> Path:
    """
    Resolve paths relative to current working directory (project root) by default.
    This matches how users run training/scripts, and how filelists are authored.
    """
    p = Path(maybe_path)
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def parse_filelist(filelist_path: Path, split_char: str = "|") -> List[Tuple[str, ...]]:
    with open(filelist_path, encoding="utf-8") as f:
        return [tuple(line.strip().split(split_char)) for line in f if line.strip()]


def compute_and_save_mel(
    wav_path: Path,
    out_path: Path,
    sample_rate: int,
    mel_mean: float,
    mel_std: float,
    mel_extractor,  # <-- Now a function
) -> Tuple[bool, str, int]:
    """
    Compute and save normalized mel spectrogram using a backend-agnostic extractor.

    Returns:
        Tuple of (success: bool, error_msg: str, mel_length: int)
    """
    try:
        audio, sr = ta.load(str(wav_path))
    except Exception as e:  # pylint: disable=broad-except
        return False, f"Failed loading {wav_path}: {e}", 0

    if sr != sample_rate:
        return False, f"Sample rate mismatch for {wav_path} (found {sr}, expected {sample_rate})", 0

    try:
        mel = mel_extractor(audio).squeeze().cpu()
        mel = normalize(mel, mel_mean, mel_std).cpu()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        arr = mel.numpy().astype(np.float32)
        np.save(out_path, arr)
        mel_length = arr.shape[-1]
        return True, "", mel_length
    except Exception as e:  # pylint: disable=broad-except
        return False, f"Processing failed for {wav_path}: {e}", 0


def compute_and_save_f0(
    wav_path: Path,
    out_path: Path,
    sample_rate: int,
    hop_length: int,
    f0_fmin: float,
    f0_fmax: float,
    f0_mean: float,
    f0_std: float,
    expected_len: int,
) -> Tuple[bool, str, int]:
    """
    Compute F0 exactly as TextMelDataset.get_f0() does, aligned to mel length.

    Args:
        expected_len: The mel spectrogram length to align F0 to

    Returns:
        Tuple of (success: bool, error_msg: str, frame_length: int)
    """
    try:
        audio, sr = ta.load(str(wav_path))
    except Exception as e:  # pylint: disable=broad-except
        return False, f"Failed loading {wav_path}: {e}", 0

    if sr != sample_rate:
        return False, f"Sample rate mismatch for {wav_path} (found {sr}, expected {sample_rate})", 0

    try:
        # Ensure mono
        if audio.dim() == 2 and audio.size(0) > 1:
            audio = audio[:1, :]

        frame_time = hop_length / float(sample_rate)
        # choose a small odd median smoothing window (in frames) to avoid exceeding available frames
        win_len = int(min(5, max(1, int(expected_len))))
        if win_len % 2 == 0:
            win_len = max(1, win_len - 1)

        # detect_pitch_frequency returns (channels, frames)
        f0 = ta.functional.detect_pitch_frequency(
            audio,
            sample_rate,
            frame_time=frame_time,
            win_length=win_len,
            freq_low=f0_fmin,
            freq_high=f0_fmax,
        )[0]

        T = f0.shape[-1]
        if T < expected_len:
            pad = torch.zeros(expected_len - T, dtype=f0.dtype)
            f0 = torch.cat([f0, pad], dim=-1)
        elif T > expected_len:
            f0 = f0[:expected_len]

        f0 = f0.unsqueeze(0)  # Add channel dimension: (1, T)

        # Normalize F0
        # Only normalize non-zero values (voiced frames)
        f0_voiced = f0[f0 > 0]
        if len(f0_voiced) > 0:
            f0_normalized = torch.where(
                f0 > 0,
                (f0 - f0_mean) / f0_std,
                torch.zeros_like(f0)
            )
        else:
            f0_normalized = f0  # all zeros, keep as is

        out_path.parent.mkdir(parents=True, exist_ok=True)
        arr = f0_normalized.cpu().numpy().astype(np.float32)
        np.save(out_path, arr)
        return True, "", f0.shape[-1]
    except Exception as e:  # pylint: disable=broad-except
        return False, f"Processing failed for {wav_path}: {e}", 0


def main():
    parser = argparse.ArgumentParser(
        description="Precompute normalized mel-spectrograms and F0 (.npy) using parameters from a dataset YAML config."
    )
    parser.add_argument(
        "-i",
        "--data-config",
        required=True,
        help="Path to data YAML (e.g., configs/data/corpus-small.yaml)",
    )
    args = parser.parse_args()

    cfg_path = Path(args.data_config).resolve()
    cfg_dir = cfg_path.parent

    cfg = _load_yaml_config(cfg_path)

    # Expected keys in the data YAML
    # train_filelist_path / valid_filelist_path
    # sample_rate, n_fft, n_feats, hop_length, win_length, f_min, f_max
    # data_statistics: { mel_mean, mel_std, f0_mean, f0_std }
    # mel_dir, f0_dir (outputs)
    required_top_keys = [
        "train_filelist_path",
        "valid_filelist_path",
        "sample_rate",
        "n_fft",
        "n_feats",
        "hop_length",
        "win_length",
        "f_min",
        "f_max",
        "data_statistics",
        "mel_dir",
    ]
    for k in required_top_keys:
        if k not in cfg:
            raise KeyError(f"Missing required key '{k}' in {cfg_path}")

    data_stats = cfg.get("data_statistics") or {}
    if "mel_mean" not in data_stats or "mel_std" not in data_stats:
        raise KeyError(f"Missing data_statistics.mel_mean or data_statistics.mel_std in {cfg_path}")

    # Resolve paths relative to current working directory (project root)
    train_filelist = _resolve_path(str(cfg["train_filelist_path"]))
    valid_filelist = _resolve_path(str(cfg["valid_filelist_path"]))
    mel_dir = _resolve_path(str(cfg["mel_dir"]))
    f0_dir = _resolve_path(str(cfg["f0_dir"])) if "f0_dir" in cfg else None

    sample_rate = int(cfg["sample_rate"])
    n_fft = int(cfg["n_fft"])
    n_mels = int(cfg["n_feats"])  # YAML uses n_feats for mel bins
    hop_length = int(cfg["hop_length"])
    win_length = int(cfg["win_length"])
    f_min = float(cfg["f_min"])
    f_max = float(cfg["f_max"])
    mel_mean = float(data_stats["mel_mean"])
    mel_std = float(data_stats["mel_std"])

    # F0 parameters (optional, with defaults)
    f0_fmin = float(cfg.get("f0_fmin", 50.0))
    f0_fmax = float(cfg.get("f0_fmax", 1100.0))
    f0_mean = float(data_stats.get("f0_mean", 0.0))
    f0_std = float(data_stats.get("f0_std", 1.0))

    # ---- NEW: read mel_backend for extractor selection ----
    mel_backend = cfg.get("mel_backend", "hifigan")
    # Instantiate the backend-agnostic mel extractor here, once
    mel_extractor = get_mel_extractor(
        mel_backend,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        n_mels=n_mels,
        f_min=f_min,
        f_max=f_max,
    )
    print(f"[precompute_corpus] Using mel_backend: {mel_backend} "
          f"with params: sample_rate={sample_rate}, n_fft={n_fft}, hop_length={hop_length}, "
          f"win_length={win_length}, n_mels={n_mels}, f_min={f_min}, f_max={f_max}")

    # Gather unique wavs from the train + valid filelists
    wavs: List[Path] = []
    for fl in [train_filelist, valid_filelist]:
        if not fl.exists():
            raise FileNotFoundError(f"Filelist not found: {fl}")
        entries = parse_filelist(fl)
        for parts in entries:
            if not parts:
                continue
            wavs.append(_resolve_path(parts[0]))

    # Deduplicate while keeping order
    seen = set()
    unique_wavs: List[Path] = []
    for w in wavs:
        if w not in seen:
            unique_wavs.append(w)
            seen.add(w)

    total = len(unique_wavs)
    ok = 0
    failures = []

    mel_dir.mkdir(parents=True, exist_ok=True)
    print(f"[precompute_corpus] Config: {cfg_path}")
    print(f"[precompute_corpus] Output: {mel_dir}")
    print(f"[precompute_corpus] Files: {total} (train+valid)")

    for i, wav_path in enumerate(unique_wavs, start=1):
        stem = wav_path.stem
        out_path = mel_dir / f"{stem}.npy"
        success, msg, mel_length = compute_and_save_mel(
            wav_path=wav_path,
            out_path=out_path,
            sample_rate=sample_rate,
            mel_mean=mel_mean,
            mel_std=mel_std,
            mel_extractor=mel_extractor,  # Pass backend-agnostic extractor
        )
        if success:
            ok += 1
            print(f"\r[precompute_corpus] {i}/{total} done.", end="", flush=True)
        else:
            print(f"[precompute_corpus] ERROR: {msg}")
            failures.append((wav_path.as_posix(), msg))

    # Write metadata for traceability
    meta = {
        "data_config": str(cfg_path),
        "sample_rate": sample_rate,
        "n_fft": n_fft,
        "n_mels": n_mels,
        "hop_length": hop_length,
        "win_length": win_length,
        "f_min": f_min,
        "f_max": f_max,
        "mel_backend": mel_backend,  # <--- NEW: record mel backend used
        "mel_extractor_params": {
            "sample_rate": sample_rate,
            "n_fft": n_fft,
            "n_mels": n_mels,
            "hop_length": hop_length,
            "win_length": win_length,
            "f_min": f_min,
            "f_max": f_max,
        },
        "mel_mean": mel_mean,
        "mel_std": mel_std,
        "num_files": total,
        "num_ok": ok,
        "num_fail": len(failures),
        "filelists": [str(train_filelist), str(valid_filelist)],
    }
    with open(mel_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    if failures:
        with open(mel_dir / "failures.txt", "w", encoding="utf-8") as f:
            for wav_path, msg in failures:
                f.write(f"{wav_path}\t{msg}\n")

    print(f"\n[precompute_corpus] Finished. ok={ok}, fail={len(failures)}.\nMetadata at {mel_dir/'metadata.json'}")

    # Compute F0 if f0_dir is specified
    if f0_dir is not None:
        print(f"\n[precompute_f0] Config: {cfg_path}")
        print(f"[precompute_f0] Output: {f0_dir}")
        print(f"[precompute_f0] Files: {total} (train+valid)")
        print(f"[precompute_f0] F0 params: fmin={f0_fmin}, fmax={f0_fmax}")
        print(f"[precompute_f0] F0 stats (for normalization): mean={f0_mean}, std={f0_std}")


        f0_dir.mkdir(parents=True, exist_ok=True)
        f0_ok = 0
        f0_failures = []

        for i, wav_path in enumerate(unique_wavs, start=1):
            stem = wav_path.stem
            mel_npy_path = mel_dir / f"{stem}.npy"

            # Load the corresponding mel to get its length
            if not mel_npy_path.exists():
                print(f"[precompute_f0] ERROR: Mel not found for {wav_path}")
                f0_failures.append((wav_path.as_posix(), "Corresponding mel file not found"))
                continue

            mel_arr = np.load(mel_npy_path)
            mel_length = mel_arr.shape[-1]

            out_path = f0_dir / f"{stem}.npy"
            success, msg, _ = compute_and_save_f0(
                wav_path=wav_path,
                out_path=out_path,
                sample_rate=sample_rate,
                hop_length=hop_length,
                f0_fmin=f0_fmin,
                f0_fmax=f0_fmax,
                f0_mean=f0_mean,
                f0_std=f0_std,
                expected_len=mel_length,
            )
            if success:
                f0_ok += 1
                print(f"\r[precompute_f0] {i}/{total} done.", end="", flush=True)
            else:
                print(f"[precompute_f0] ERROR: {msg}")
                f0_failures.append((wav_path.as_posix(), msg))

        # Write F0 metadata
        f0_meta = {
            "data_config": str(cfg_path),
            "sample_rate": sample_rate,
            "hop_length": hop_length,
            "f0_fmin": f0_fmin,
            "f0_fmax": f0_fmax,
            "f0_mean": f0_mean,
            "f0_std": f0_std,
            "num_files": total,
            "num_ok": f0_ok,
            "num_fail": len(f0_failures),
            "filelists": [str(train_filelist), str(valid_filelist)],
        }
        with open(f0_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(f0_meta, f, indent=2)

        if f0_failures:
            with open(f0_dir / "failures.txt", "w", encoding="utf-8") as f:
                for wav_path, msg in f0_failures:
                    f.write(f"{wav_path}\t{msg}\n")

        print(f"\n[precompute_f0] Finished. ok={f0_ok}, fail={len(f0_failures)}.\nMetadata at {f0_dir/'metadata.json'}")


if __name__ == "__main__":
    main()
