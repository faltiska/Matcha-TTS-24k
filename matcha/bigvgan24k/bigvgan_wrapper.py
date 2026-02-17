import torch
import json
import bigvgan
from huggingface_hub import hf_hub_download

# BigVGAN expects a namespace/object for hyperparameters
class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

class BigVGANWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, mel):
        # BigVGAN v2 expects [B, 100, T]
        # returns [B, 1, Samples]
        with torch.inference_mode():
            wav = self.model(mel)
        return wav.squeeze(1)

def load_bigvgan(device="cuda"):
    from pathlib import Path
    
    wrapper_dir = Path(__file__).parent
    config_path = wrapper_dir / "config.json"
    checkpoint_path = wrapper_dir / "bigvgan_generator.pt"

    print(f"Loading BigVGAN v2 from local files...")

    # Load the config and convert to AttrDict
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    h = AttrDict(config_dict)

    # Instantiate the model manually
    model = bigvgan.BigVGAN(h, use_cuda_kernel=False)

    # Load the weights
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(state_dict['generator'])

    model.remove_weight_norm()
    model.eval().to(device)

    return BigVGANWrapper(model)


if __name__ == "__main__":
    """
    Test BigVGAN vocoder quality by computing MCD between ground truth and vocoder-generated audio.
    
    For each speaker in the corpus, randomly selects one validation sample, loads the wav file,
    generates mel spectrogram using BigVGAN's method, generates audio using BigVGAN, and compares 
    it to the original wav file using the Mel Cepstral Distortion (MCD) metric.
    
    Lower MCD values indicate better vocoder quality:
      < 5.0 dB  = Excellent
      5-7 dB    = Good
      7-10 dB   = Fair
      > 10 dB   = Poor
    """
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="pyworld")
    
    import torch
    import torchaudio as ta
    import yaml
    import random
    import time
    from pathlib import Path
    from pymcd.mcd import Calculate_MCD
    from matcha.bigvgan24k.meldataset import get_mel_spectrogram

    cfg_path = Path("configs/data/corpus-small-24k.yaml")
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    
    sample_rate = config["sample_rate"]
    n_spks = config["n_spks"]
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
    
    vocoder = load_bigvgan()
    mcd_toolbox = Calculate_MCD(MCD_mode="dtw")
    results = []
    vocoder_times = []
    out_dir = Path("bigvgan_test_output")
    out_dir.mkdir(exist_ok=True)
    
    for spk, audio_rel_path in samples:
        wav_path = Path("data/corpus-small-24k/wav") / f"{audio_rel_path}.wav"
        
        # Load wav using torchaudio
        waveform, sr = ta.load(str(wav_path))
        waveform = waveform.cuda()
        
        # Compute mel using BigVGAN's method
        mel = get_mel_spectrogram(waveform, vocoder.model.h)
        
        # Time only the vocoder inference
        torch.cuda.synchronize()
        start = time.time()
        with torch.no_grad():
            y_hat = vocoder(mel).squeeze()
        torch.cuda.synchronize()
        vocoder_time = time.time() - start
        vocoder_times.append(vocoder_time)
        
        gen_path = out_dir / f"spk{spk}_{wav_path.stem}_bigvgan.wav"
        ta.save(str(gen_path), torch.clamp(y_hat, -1.0, 1.0).cpu().unsqueeze(0), sample_rate)
        
        mcd_value = mcd_toolbox.calculate_mcd(str(gen_path), str(wav_path))
        results.append(mcd_value)
        audio_duration = waveform.shape[1] / sample_rate
        rtf = vocoder_time / audio_duration
        print(f"Speaker {spk} {wav_path.name:30s} MCD: {mcd_value:6.2f} dB | Vocoder: {vocoder_time:.3f}s | RTF: {rtf:.3f}")
    
    print("-" * 70)
    if results:
        print(f"{'Average':40s} MCD: {sum(results)/len(results):6.2f} dB")
    if vocoder_times:
        print(f"{'Average vocoder time':40s} {sum(vocoder_times)/len(vocoder_times):.3f}s")