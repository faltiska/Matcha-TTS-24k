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
    # Subjective test for Vocos-24k (waveform->mel->waveform)
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

    parser = argparse.ArgumentParser(description="Test Vocos-24k vocoder on a WAV file")
    parser.add_argument("--wav", type=str, required=True, help="Path to input WAV file")
    default_out = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "vocoder-test24k.wav"))
    parser.add_argument("--out", type=str, default=default_out, help="Output WAV path (default: project_root/vocoder-test24k.wav)")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda", help="Device to run on")
    # Mel/STFT params (24 kHz defaults)
    parser.add_argument("--sr", type=int, default=24000, help="Sampling rate for mel and output (default: 24000)")

    args = parser.parse_args()

    print(f"[vocos24k-test] device={args.device}")

    # Load audio at desired SR (matches mel config)
    y, _ = librosa.load(args.wav, sr=args.sr)
    y = librosa_normalize(y) * 0.95 # matches what we do during training
    y_t = torch.from_numpy(y).float().unsqueeze(0).to(args.device)

    # Load Vocos and infer using the same mel extractor as training (factory: vocos24k)
    vocoder = load_model(device=args.device)
    vocoder.eval()
    with torch.no_grad():
        mel_extractor = get_mel_extractor("vocos24k")
        mel = mel_extractor(y_t)
        print(f"[vocos24k-test] mel (factory: vocos24k) shape: {tuple(mel.shape)}")
        y_hat = vocoder(mel).squeeze(0)  # [T] or [1, T] -> [T]
        if y_hat.dim() > 1:
            y_hat = y_hat.squeeze(0)

    # Save WAV (16-bit PCM) at requested SR
    y_np = y_hat.detach().float().cpu().numpy()
    y_np = np.clip(y_np, -1.0, 1.0)
    wav_int16 = (y_np * MAX_WAV_VALUE).astype(np.int16)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    wav_write(args.out, args.sr, wav_int16)
    print(f"[vocos24k-test] Reconstituted file saved to: {args.out}")
