import datetime as dt
import io
import subprocess
import time
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import soundfile as sf

from matcha.cli import load_matcha, load_vocoder, process_text, to_waveform

CHECKPOINT_PATH = "logs/train/corpus-small-24k/runs/2026-01-20_09-04-05/checkpoints/checkpoint_epoch=159.ckpt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
vocoder = None
denoiser = None

VOICES = [
    {"id": "0", "lang": "en-us", "gender": "male",   "name": "Kai"},
    {"id": "1", "lang": "en-us", "gender": "female", "name": "Jane"},
    {"id": "2", "lang": "en-us", "gender": "female", "name": "Aria"},
    {"id": "3", "lang": "en-gb", "gender": "female", "name": "Bella"},
    {"id": "4", "lang": "en-gb", "gender": "male",   "name": "Brian"},
    {"id": "5", "lang": "en-gb", "gender": "male",   "name": "Arthur"},
    {"id": "6", "lang": "en-us", "gender": "female", "name": "Nicole"},
    {"id": "7", "lang": "ro",    "gender": "male",   "name": "Emil"},
    {"id": "8", "lang": "fr-fr", "gender": "female", "name": "Denise"},
    {"id": "9", "lang": "fr-fr", "gender": "male",   "name": "Henri"},
]

app = FastAPI(title="Matcha-TTS Inference Server")

class InferenceRequest(BaseModel):
    input: str
    voice: int = 0
    response_format: str = "mp3"
    speed: float = 1.0
    steps: int = 20
    voice_mix: dict[int, float] | None = None # {0: 0.1, 2: 0.9}

def convert_to_mp3(waveform, sample_rate):
    wav_buffer = io.BytesIO()
    sf.write(wav_buffer, waveform, sample_rate, format="WAV", subtype="PCM_16")
    wav_buffer.seek(0)
    
    process = subprocess.Popen(
        ["ffmpeg", "-i", "pipe:0", "-f", "mp3", "-ab", "192k", "pipe:1"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL
    )
    mp3_data, _ = process.communicate(input=wav_buffer.read())
    return mp3_data

@app.on_event("startup")
def load_models():
    global model, vocoder, denoiser
    print(f"[🍵] Loading model from {CHECKPOINT_PATH}")
    model = load_matcha("custom_model", CHECKPOINT_PATH, DEVICE)
    model.decoder.solver = "midpoint"

    if not hasattr(model, "sample_rate"):
        model.sample_rate = 24000
    if not hasattr(model, "hop_length"):
        model.hop_length = 256

    vocoder_path = Path.home() / ".local/share/matcha_tts/vocos"
    vocoder, denoiser = load_vocoder("vocos", vocoder_path, DEVICE)
    print("[🍵] Models loaded successfully")

@app.get("/")
def root():
    return {"status": "ok", "message": "Matcha-TTS server is running"}

@app.get("/api/v1/speak")
@app.get("/prod/speak/evie")
@app.get("/test/speak/evie")
def get_voices():
    return VOICES

@app.post("/api/v1/speak")
@app.post("/prod/speak/evie")
@app.post("/test/speak/evie")
@torch.inference_mode()
def synthesize(request: InferenceRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = request.input.strip()
    length_scale = request.speed

    language = VOICES[request.voice]["lang"]

    if request.voice_mix:
        spk = torch.zeros(1, model.spk_emb_dim, device=DEVICE)
        for spk_id, weight in request.voice_mix.items():
            spk += weight * model.spk_emb(torch.tensor([spk_id], device=DEVICE, dtype=torch.long))
    else:
        spk = torch.tensor([request.voice], device=DEVICE, dtype=torch.long)

    text_processed = process_text(1, text, language, DEVICE)

    start_t = dt.datetime.now()
    output = model.synthesise(
        text_processed["x"],
        text_processed["x_lengths"],
        n_timesteps=request.steps,
        temperature=0.8,
        spks=spk,
        length_scale=length_scale,
    )
    output["waveform"] = to_waveform(output["mel"], vocoder, denoiser, 0.00025)

    t = (dt.datetime.now() - start_t).total_seconds()
    sample_rate = getattr(model, "sample_rate")
    rtf = t * sample_rate / output["waveform"].shape[-1]

    print(f"[🍵] Inference time: {t:.2f}s, RTF: {rtf:.4f}")

    waveform = output["waveform"].cpu().numpy()
    mp3_data = convert_to_mp3(waveform, sample_rate)
    return Response(content=mp3_data, media_type="audio/mpeg")

# Run it with python -m matcha.server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("matcha.server:app", host="0.0.0.0", port=8000, reload=True)
