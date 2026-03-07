"""
Add a new speaker to an existing MatchaTTS checkpoint using a trained StyleEncoder.

Encodes reference mel spectrograms from the new speaker, predicts the 3 speaker embedding
vectors, appends them to the checkpoint's embedding tables, increments n_spks, and saves
a new checkpoint. The new speaker gets ID = original n_spks.

Usage:
    python -m matcha.utils.add_speaker \
        --matcha_ckpt logs/train/v6/checkpoint.ckpt \
        --style_encoder_ckpt logs/style_encoder.pt \
        --mel_dir data/new-speaker/mels \
        --output_ckpt logs/train/v6/checkpoint_with_new_speaker.ckpt
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from matcha.models.style_encoder import StyleEncoder
from matcha.utils.model import normalize

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_style_encoder(encoder_path: str) -> StyleEncoder:
    ckpt = torch.load(encoder_path, map_location=DEVICE, weights_only=False)
    hparams = ckpt["hyper_parameters"]

    # Reconstruct the matcha checkpoint hparams to get embedding dims
    matcha = torch.load(hparams["matcha_ckpt"], map_location="cpu", weights_only=False)
    matcha_hparams = matcha["hyper_parameters"]
    n_feats          = matcha_hparams["n_feats"]
    spk_emb_dim_enc  = matcha_hparams.get("spk_emb_dim_enc") or matcha_hparams["spk_emb_dim"]
    spk_emb_dim_dur  = matcha_hparams.get("spk_emb_dim_dur") or matcha_hparams["spk_emb_dim"]
    spk_emb_dim_dec  = matcha_hparams.get("spk_emb_dim_dec") or matcha_hparams["spk_emb_dim"]

    model = StyleEncoder(
        n_feats=n_feats,
        spk_emb_dim_enc=spk_emb_dim_enc,
        spk_emb_dim_dur=spk_emb_dim_dur,
        spk_emb_dim_dec=spk_emb_dim_dec,
        n_channels=hparams["n_channels"],
    ).to(DEVICE)

    # Lightning prefixes encoder weights with "encoder."
    encoder_sd = {k.removeprefix("encoder."): v for k, v in ckpt["state_dict"].items() if k.startswith("encoder.")}
    model.load_state_dict(encoder_sd)
    model.eval()
    return model


@torch.inference_mode()
def predict_speaker_embeddings(style_encoder, mel_dir: str, mel_mean: float, mel_std: float):
    mel_paths = sorted(Path(mel_dir).rglob("*.npy"))
    if not mel_paths:
        raise FileNotFoundError(f"No .npy mel files found in {mel_dir}")
    print(f"[+] Encoding {len(mel_paths)} reference mels from {mel_dir}")

    mels = []
    for path in mel_paths:
        arr = torch.from_numpy(np.load(path).astype("float32")).to(DEVICE)
        arr = normalize(arr, mel_mean, mel_std)
        mels.append(arr)

    lengths = torch.tensor([m.shape[-1] for m in mels], dtype=torch.long, device=DEVICE)
    padded = torch.nn.utils.rnn.pad_sequence(
        [m.T for m in mels], batch_first=True
    ).permute(0, 2, 1)  # (N, n_feats, T_max)

    enc_emb, dur_emb, dec_emb = style_encoder(padded, lengths)
    return enc_emb.cpu(), dur_emb.cpu(), dec_emb.cpu()


def add_speaker(args):
    print(f"[+] Loading MatchaTTS checkpoint from {args.matcha_ckpt}")
    ckpt = torch.load(args.matcha_ckpt, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    hparams = ckpt["hyper_parameters"]

    mel_mean = hparams["data_statistics"]["mel_mean"]
    mel_std = hparams["data_statistics"]["mel_std"]
    old_n_spks = hparams["n_spks"]
    new_speaker_id = old_n_spks
    print(f"[+] Current n_spks={old_n_spks}, new speaker will be ID {new_speaker_id}")

    style_encoder = load_style_encoder(args.style_encoder_ckpt)
    enc_emb, dur_emb, dec_emb = predict_speaker_embeddings(style_encoder, args.mel_dir, mel_mean, mel_std)

    embedding_keys = {
        "encoder_speaker_embeddings.weight": enc_emb,
        "duration_speaker_embeddings.weight": dur_emb,
        "decoder_speaker_embeddings.weight": dec_emb,
    }
    for key, new_row in embedding_keys.items():
        sd[key] = torch.cat([sd[key], new_row], dim=0)

    hparams["n_spks"] = old_n_spks + 1

    output_path = Path(args.output_ckpt)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, output_path)
    print(f"[+] Saved new checkpoint with speaker {new_speaker_id} to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--matcha_ckpt", required=True, help="Path to the source MatchaTTS checkpoint")
    parser.add_argument("--style_encoder_ckpt", required=True, help="Path to the trained StyleEncoder checkpoint")
    parser.add_argument("--mel_dir", required=True, help="Directory containing .npy mel files for the new speaker")
    parser.add_argument("--output_ckpt", required=True, help="Path to save the updated checkpoint")
    args = parser.parse_args()
    add_speaker(args)
