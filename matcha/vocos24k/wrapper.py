import torch

class VocosWrapper(torch.nn.Module):
    def __init__(self, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = model
        
    def forward(self, mel):
        return self.model.decode(mel)
    
def load_model(device="cuda"):
    from vocos import Vocos
    model = Vocos.from_pretrained("charactr/vocos-mel-24khz")
    model.eval().to(device)
    return VocosWrapper(model)


if __name__ == "__main__":
    # Subjective test for Vocos-24k (waveform->mel->waveform or MEL->waveform)
    import argparse
    import os

    import numpy as np
    import torch
    from scipy.io.wavfile import write as wav_write

    try:
        import librosa
        from librosa.util import normalize as librosa_normalize
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "librosa is required for this test script. Please install it (pip install librosa)."
        ) from e

    from matcha.utils.audio import MAX_WAV_VALUE
    from matcha.mel.extractors import get_mel_extractor

    parser = argparse.ArgumentParser(description="Test Vocos-24k vocoder: (1) wav->mel->wav or (2) mel.npy->wav")
    parser.add_argument("--wav", type=str, help="Path to input WAV file (used if --mel not provided)")
    parser.add_argument("--mel", type=str, default=None, help="Optional path to a precomputed mel numpy file (.npy). If given, skips wav audio and reconstructs from this numpy array (shape: [n_mels, T] or [1, n_mels, T]).")
    parser.add_argument("--data-config", "-c", required=True, type=str, help="Path to corpus data YAML (as used by precompute_corpus). All params (mel_mean, mel_std, sample_rate, n_feats) ARE ALWAYS TAKEN FROM THIS FILE.")
    default_out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vocoder-test24k.wav"))
    parser.add_argument("--out", type=str, default=default_out, help="Output WAV path (default: project_root/vocoder-test24k.wav)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda", help="Device to run on")

    args = parser.parse_args()
    print(f"[vocos24k-test] device={args.device}")

    # --- All core params come ONLY from YAML config ---
    import yaml
    from pathlib import Path
    cfg_path = Path(args.data_config).resolve()
    with open(cfg_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    print(f"[vocos24k-test] Loaded data config: {cfg_path}")
    try:
        mel_mean = float(config["data_statistics"]["mel_mean"])
        mel_std = float(config["data_statistics"]["mel_std"])
        sample_rate = int(config["sample_rate"])
        n_feats = int(config["n_feats"])
    except Exception as e:
        raise RuntimeError(f"Config file missing required fields. Check for: sample_rate, n_feats, data_statistics.mel_mean, data_statistics.mel_std\n{e}")

    print(f"[vocos24k-test] Effective parameters: mel_mean={mel_mean}, mel_std={mel_std}, sample_rate={sample_rate}, n_feats={n_feats}")

    vocoder = load_model(device=args.device)
    vocoder.eval()

    # Determine source of mel: from file or via extraction from wav
    # Require normalization params in all paths!
    if (mel_mean is None) or (mel_std is None):
        raise ValueError("Both --mel-mean and --mel-std (or --data-config with data_statistics) must be provided!")

    if args.mel is not None:
        if not os.path.exists(args.mel):
            raise FileNotFoundError(f"Specified mel file does not exist: {args.mel}")
        print(f"[vocos24k-test] Loading mel from npy: {args.mel}")
        mel = np.load(args.mel)
        if mel.ndim == 2:
            mel = mel[np.newaxis, ...]  # ensure shape [1, n_mels, T]
        # Denormalize
        print(f"[vocos24k-test] Denormalizing mel using mean={mel_mean}, std={mel_std}")
        mel = mel * mel_std + mel_mean
        mel = torch.from_numpy(mel).float().to(args.device)
        print(f"[vocos24k-test] Loaded mel shape: {tuple(mel.shape)}")
    elif args.wav is not None:
        # Load, normalize, extract mel
        y, _ = librosa.load(args.wav, sr=sample_rate)
        y = librosa_normalize(y) * 0.95  # matches what we do during training
        y_t = torch.from_numpy(y).float().unsqueeze(0).to(args.device)
        mel_extractor = get_mel_extractor("vocos24k", sample_rate=sample_rate, n_mels=n_feats)
        mel = mel_extractor(y_t).cpu().numpy()
        print(f"[vocos24k-test] Extracted mel (factory: vocos24k) shape: {tuple(mel.shape)}")
        # Normalize (match precompute_corpus)
        print(f"[vocos24k-test] Normalizing mel with mean={mel_mean}, std={mel_std}")
        mel = (mel - mel_mean) / mel_std
        # Then denormalize before vocoder (symmetry with --mel path)
        print(f"[vocos24k-test] Denormalizing mel using mean={mel_mean}, std={mel_std} (restores to vocoder expected domain)")
        mel = mel * mel_std + mel_mean
        mel = torch.from_numpy(mel).float().to(args.device)
        print(f"[vocos24k-test] Final (denormalized) mel shape: {tuple(mel.shape)}")
    else:
        raise ValueError("You must provide either --wav or --mel")

    # Vocoder inference
    with torch.no_grad():
        y_hat = vocoder(mel).squeeze(0)  # [T] or [1, T] -> [T]
        if y_hat.dim() > 1:
            y_hat = y_hat.squeeze(0)

    # Save WAV (16-bit PCM) at requested SR
    y_np = y_hat.detach().float().cpu().numpy()
    y_np = np.clip(y_np, -1.0, 1.0)
    wav_int16 = (y_np * MAX_WAV_VALUE).astype(np.int16)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    wav_write(args.out, sample_rate, wav_int16)
    print(f"[vocos24k-test] Reconstituted file saved to: {args.out}")
