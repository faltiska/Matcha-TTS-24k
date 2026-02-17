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
    """
    Test Vocos vocoder quality by computing MCD between ground truth and vocoder-generated audio.
    
    For each speaker in the corpus, randomly selects one validation sample, loads its precomputed
    mel spectrogram, generates audio using Vocos, and compares it to the original wav file using
    the Mel Cepstral Distortion (MCD) metric.
    
    Lower MCD values indicate better vocoder quality:
      < 5.0 dB  = Excellent
      5-7 dB    = Good
      7-10 dB   = Fair
      > 10 dB   = Poor
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pyworld")
    
    import csv
    import numpy as np
    import torch
    import torchaudio as ta
    import yaml
    import random
    from pathlib import Path
    from pymcd.mcd import Calculate_MCD

    cfg_path = Path("configs/data/corpus-small-24k.yaml")
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    
    mel_mean = config["data_statistics"]["mel_mean"]
    mel_std = config["data_statistics"]["mel_std"]
    sample_rate = config["sample_rate"]
    n_spks = config["n_spks"]
    mel_dir = Path(config["mel_dir"])
    val_csv = Path(config["valid_filelist_path"])
    
    # Parse CSV and group by speaker
    by_speaker = {i: [] for i in range(n_spks)}
    with open(val_csv, "r") as f:
        for line in f:
            parts = line.strip().split("|")
            if len(parts) >= 2:
                by_speaker[int(parts[1])].append(parts[0])
    
    # Pick 1 random sample per speaker
    samples = [(i, random.choice(by_speaker[i])) for i in range(n_spks)]
    
    vocoder = load_model()
    mcd_toolbox = Calculate_MCD(MCD_mode="dtw")
    results = []
    out_dir = Path("vocos_test_output")
    out_dir.mkdir(exist_ok=True)
    
    for spk, audio_rel_path in samples:
        wav_path = Path("data/corpus-small-24k/wav") / f"{audio_rel_path}.wav"
        mel_path = mel_dir / f"{audio_rel_path}.npy"
        
        mel = np.load(mel_path)
        if mel.ndim == 2:
            mel = mel[np.newaxis, ...]
        mel = torch.from_numpy(mel * mel_std + mel_mean).float().cuda()
        
        with torch.no_grad():
            y_hat = vocoder(mel).squeeze()
        
        gen_path = out_dir / f"spk{spk}_{wav_path.stem}_vocos.wav"
        ta.save(str(gen_path), torch.clamp(y_hat, -1.0, 1.0).cpu().unsqueeze(0), sample_rate)
        
        mcd_value = mcd_toolbox.calculate_mcd(str(gen_path), str(wav_path))
        results.append(mcd_value)
        print(f"Speaker {spk} {wav_path.name:30s} MCD: {mcd_value:6.2f} dB")
    
    print("-" * 70)
    if results:
        print(f"{'Average':40s} MCD: {sum(results)/len(results):6.2f} dB")
