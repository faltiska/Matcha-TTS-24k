r"""
The file creates a pickle file where the values needed for loading of dataset is stored and the model can load it
when needed.

Parameters from hparam.py will be used
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict, Any

import rootutils
import torch
import torchaudio as ta
import numpy as np
from hydra import compose, initialize
from omegaconf import open_dict
from tqdm.auto import tqdm

from matcha.mel.extractors import get_mel_extractor
from matcha.utils.logging_utils import pylogger

log = pylogger.get_pylogger(__name__)


def parse_filelist(filelist_path: Path, split_char: str = "|") -> List[Tuple[str, ...]]:
    """Parse a filelist CSV file."""
    with open(filelist_path, encoding="utf-8") as f:
        return [tuple(line.strip().split(split_char)) for line in f if line.strip()]


def compute_and_load_mel(
    wav_path: Path,
    sample_rate: int,
    mel_extractor,
) -> Tuple[bool, torch.Tensor, str]:
    """
    Load wav and compute mel spectrogram (raw, not normalized).
    
    Returns:
        Tuple of (success: bool, mel_tensor: torch.Tensor, error_msg: str)
    """
    try:
        audio, sr = ta.load(str(wav_path))
    except Exception as e:
        return False, None, f"Failed loading {wav_path}: {e}"

    if sr != sample_rate:
        return False, None, f"Sample rate mismatch for {wav_path} (found {sr}, expected {sample_rate})"

    try:
        mel = mel_extractor(audio).squeeze().cpu()
        
        if np.isnan(mel.numpy()).any() or np.isinf(mel.numpy()).any():
            return False, None, f"NaN/Inf detected in mel for {wav_path}"
        
        return True, mel, ""
    except Exception as e:
        return False, None, f"Processing failed for {wav_path}: {e}"


def compute_data_statistics(
    train_filelist_path: str,
    valid_filelist_path: str,
    sample_rate: int,
    n_fft: int,
    n_mels: int,
    hop_length: int,
    win_length: int,
    f_min: float,
    f_max: float,
    mel_backend: str = "vocos",
) -> Dict[str, float]:
    """
    Compute mel statistics by directly loading all wav files (no DataLoader).
    Catches and reports all file loading errors.
    Statistics are computed on raw mel spectrograms before normalization.
    
    Returns:
        Dict with mel_mean and mel_std
    """
    # Setup paths
    train_filelist = Path(train_filelist_path)
    valid_filelist = Path(valid_filelist_path)
    filelist_dir = train_filelist.parent
    
    # Initialize mel extractor
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
    
    # Gather unique wav files from train and valid filelists
    rel_and_abs_wavs: List[Tuple[str, Path]] = []
    for fl in [train_filelist, valid_filelist]:
        if not fl.exists():
            raise FileNotFoundError(f"Filelist not found: {fl}")
        entries = parse_filelist(fl)
        for parts in entries:
            if not parts:
                continue
            # Extract relative base path (no extension)
            rel_base = parts[0]
            wav_path = (filelist_dir / "wav" / (rel_base + ".wav")).resolve()
            rel_and_abs_wavs.append((rel_base, wav_path))
    
    total = len(rel_and_abs_wavs)
    total_mel_sum = 0.0
    total_mel_sq_sum = 0.0
    total_mel_len = 0
    ok = 0
    failures = []
    
    print(f"Computing statistics from {total} wav files...")
    
    for i, (rel_base, wav_path) in enumerate(rel_and_abs_wavs, start=1):
        success, mel, error_msg = compute_and_load_mel(
            wav_path=wav_path,
            sample_rate=sample_rate,
            mel_extractor=mel_extractor,
        )
        
        if success:
            total_mel_sum += torch.sum(mel).item()
            total_mel_sq_sum += torch.sum(torch.pow(mel, 2)).item()
            total_mel_len += mel.shape[-1]
            ok += 1
            print(f"\r[Statistics] {i}/{total} processed...", end="", flush=True)
        else:
            print(f"\n[Statistics] ERROR: {error_msg}")
            failures.append((str(wav_path), error_msg))
    
    print(f"\n\nProcessed {ok} wav files successfully.")
    
    if failures:
        print(f"\n⚠️  WARNING: {len(failures)} files failed to load:")
        for i, (wav_path, error_msg) in enumerate(failures[:20], 1):
            print(f"  {i}. {wav_path}")
            print(f"     {error_msg}")
        if len(failures) > 20:
            print(f"  ... and {len(failures) - 20} more files with errors")
    
    if ok == 0:
        raise RuntimeError("No files were successfully loaded!")
    
    # Compute statistics
    data_mean = total_mel_sum / (total_mel_len * n_mels)
    data_std = np.sqrt((total_mel_sq_sum / (total_mel_len * n_mels)) - (data_mean ** 2))
    
    return {"mel_mean": round(float(data_mean), 6), "mel_std": round(float(data_std), 6)}


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-i",
        "--input-config",
        type=str,
        default="vctk.yaml",
        help="The name of the yaml config file under configs/data",
    )

    args = parser.parse_args()

    with initialize(version_base="1.3", config_path="../../configs/data"):
        cfg = compose(config_name=args.input_config, return_hydra_config=True, overrides=[])

    root_path = rootutils.find_root(search_from=__file__, indicator=".project-root")

    with open_dict(cfg):
        del cfg["hydra"]
        del cfg["_target_"]
        cfg["train_filelist_path"] = str(os.path.join(root_path, cfg["train_filelist_path"]))
        cfg["valid_filelist_path"] = str(os.path.join(root_path, cfg["valid_filelist_path"]))

    mel_dir = cfg.get("mel_dir")
    if mel_dir and os.path.exists(mel_dir):
        # Not only stats are not needed; when the mel files exist, the method get_mel  
        # will load from existing files instead of computing from wav and the stats will be wrong. 
        print(f"ERROR: Directory '{mel_dir}' already exists, will not compute statistics.")
        sys.exit(1)

    # Compute mel statistics with direct file loading (no DataLoader)
    params = compute_data_statistics(
        train_filelist_path=cfg["train_filelist_path"],
        valid_filelist_path=cfg["valid_filelist_path"],
        sample_rate=int(cfg["sample_rate"]),
        n_fft=int(cfg["n_fft"]),
        n_mels=int(cfg["n_feats"]),
        hop_length=int(cfg["hop_length"]),
        win_length=int(cfg["win_length"]),
        f_min=float(cfg["f_min"]),
        f_max=float(cfg["f_max"]),
        mel_backend=cfg.get("mel_backend", "vocos"),
    )
    
    print("\ndata_statistics:")
    print(f"  mel_mean: {params.get('mel_mean')}")
    print(f"  mel_std: {params.get('mel_std')}")

if __name__ == "__main__":
    main()
