"""
Add a new speaker to a Matcha checkpoint using a trained Style Encoder.

For each recording in the CSV, loads the precomputed mel, runs ASE and RSE to predict
the speaker embeddings. Averages predictions across all recordings and writes the new
speaker's embeddings into an expanded copy of the Matcha checkpoint.

Mel files are expected at <csv_dir>/mels/<rel_base>.npy, as produced by precompute_mels.py.

Usage:
    python -m matcha.add_speaker \
        --style-encoder-ckpt logs/style_encoder/checkpoint.ckpt \
        --matcha-ckpt logs/train/v6/checkpoint.ckpt \
        --csv data/extra-speakers-24k/train.csv \
        --output logs/train/v6/checkpoint_with_new_speaker.ckpt
"""

import argparse
import shutil
from pathlib import Path

import numpy as np
import torch

from matcha.models.style_encoder import StyleEncoderLightningModule

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def parse_csv(csv_path):
    rows = []
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 4:
                rows.append(parts)
    return rows


def predict_embeddings(style_encoder, csv_rows, mel_dir):
    embs = []

    for i, row in enumerate(csv_rows):
        rel_base = row[0]

        mel_path = Path(mel_dir) / (rel_base + ".fine.npy")
        mel = torch.from_numpy(np.load(mel_path).astype(np.float32)).to(DEVICE)
        y = mel.unsqueeze(0)

        with torch.no_grad():
            pred_spk_emb = style_encoder.acoustic_style_encoder(y, torch.ones(1, 1, y.shape[-1], device=DEVICE))

        embs.append(pred_spk_emb.squeeze(0))
        print(f"\r[add_speaker] {i + 1}/{len(csv_rows)}", end="", flush=True)

    print()
    avg_spk_emb = torch.stack(embs).mean(dim=0)
    return avg_spk_emb


def expand_embedding_table(state_dict, key, new_row):
    old_weight = state_dict[key]
    new_weight = torch.cat([old_weight, new_row.unsqueeze(0).cpu()], dim=0)
    state_dict[key] = new_weight
    return new_weight.shape[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--style-encoder-ckpt", required=True)
    parser.add_argument("--matcha-ckpt", required=True)
    parser.add_argument("--csv", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    print("[add_speaker] Loading style encoder...")
    style_encoder = StyleEncoderLightningModule.load_from_checkpoint(
        args.style_encoder_ckpt, map_location=DEVICE, weights_only=False
    )
    style_encoder.eval()
    style_encoder.to(DEVICE)

    csv_rows = parse_csv(args.csv)
    mel_dir = Path(args.csv).parent / "mels"
    print(f"[add_speaker] Processing {len(csv_rows)} recordings from {mel_dir}...")
    avg_spk_emb = predict_embeddings(style_encoder, csv_rows, mel_dir)

    output_path = Path(args.output)
    shutil.copy2(args.matcha_ckpt, output_path)
    ckpt = torch.load(output_path, map_location="cpu", weights_only=False)

    sd = ckpt["state_dict"]
    new_n_spks = expand_embedding_table(sd, "speaker_embeddings.weight", avg_spk_emb)
    ckpt["hyper_parameters"]["n_spks"] = new_n_spks

    torch.save(ckpt, output_path)
    print(f"[add_speaker] Saved to {output_path} (now {new_n_spks} speakers, new speaker id={new_n_spks - 1})")


if __name__ == "__main__":
    main()
