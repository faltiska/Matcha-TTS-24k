"""
Precompute and cache normalized mel-spectrograms for a given dataset YAML config.

This script reads all required parameters (filelists, mel params, normalization stats,
and output directories) from a Hydra-style data config YAML like configs/data/corpus-small.yaml.

It then saves:
  - Normalized mel spectrograms as .npy files (shape: (n_mels, T)) into mel_dir
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
        if np.isnan(arr).any() or np.isinf(arr).any():
            raise RuntimeError(f"[precompute_corpus] ERROR: NaN/Inf detected in mel for {out_path}")
        np.save(out_path, arr)
        mel_length = arr.shape[-1]
        return True, "", mel_length
    except Exception as e:  # pylint: disable=broad-except
        return False, f"Processing failed for {wav_path}: {e}", 0


def main():
    parser = argparse.ArgumentParser(
        description="Precompute normalized mel-spectrograms (.npy) using parameters from a dataset YAML config."
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
    # data_statistics: { mel_mean, mel_std }
    # mel_dir (output)
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

    sample_rate = int(cfg["sample_rate"])
    n_fft = int(cfg["n_fft"])
    n_mels = int(cfg["n_feats"])  # YAML uses n_feats for mel bins
    hop_length = int(cfg["hop_length"])
    win_length = int(cfg["win_length"])
    f_min = float(cfg["f_min"])
    f_max = float(cfg["f_max"])
    mel_mean = float(data_stats["mel_mean"])
    mel_std = float(data_stats["mel_std"])

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
    rel_and_abs_wavs: List[Tuple[str, Path]] = []
    for fl in [train_filelist, valid_filelist]:
        if not fl.exists():
            raise FileNotFoundError(f"Filelist not found: {fl}")
        entries = parse_filelist(fl)
        for parts in entries:
            if not parts:
                continue
            rel_base = parts[0]
            wav_path = (fl.parent / "wav" / (rel_base + ".wav")).resolve()
            rel_and_abs_wavs.append((rel_base, wav_path))

    total = len(rel_and_abs_wavs)
    ok = 0
    failures = []

    mel_dir.mkdir(parents=True, exist_ok=True)
    print(f"[precompute_corpus] Config: {cfg_path}")
    print(f"[precompute_corpus] Output: {mel_dir}")
    print(f"[precompute_corpus] Files: {total} (train+validate)")

    for i, (rel_base, wav_path) in enumerate(rel_and_abs_wavs, start=1):
        # rel_base is like "1/abc"
        out_path = mel_dir / (rel_base + ".npy")
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
            failures.append((str(wav_path), msg))

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


if __name__ == "__main__":
    main()
