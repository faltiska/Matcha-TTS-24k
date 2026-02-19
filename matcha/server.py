import datetime as dt
import io
import os
import subprocess
import time
from pathlib import Path

# Set HuggingFace cache BEFORE any imports that might use it
cache_base = Path(os.environ.get("MATCHA_CACHE_DIR", Path.cwd() / ".cache"))
os.environ["HF_HOME"] = str(cache_base / "huggingface")

from parse import parse
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

from matcha.inference import load_matcha, load_vocoder, process_text, to_waveform, post_process, convert_to_mp3, SAMPLE_RATE, ODE_SOLVER

CHECKPOINT_PATH = "logs/train/corpus-small-24k/v1/checkpoint_epoch=579.ckpt"
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", CHECKPOINT_PATH)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = None
vocoder = None

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
    voice: int | str = 0  # Voice ID or Voice Mix in the form of "2(20)+5(80)" meaning 20% of voice 2 mixed with 80% of voice 5 
    response_format: str = "mp3"
    speed: float = 1.0
    steps: int = 15


def parse_voice_mix(voice_str: str):
    """Parse voice mix string like '2(70)+6(30)' into (id1, weight1, id2, weight2)"""
    parts = voice_str.split('+')
    voice1 = parse("{:d}({:d})", parts[0])
    voice2 = parse("{:d}({:d})", parts[1])
    return voice1[0], voice1[1] / 100, voice2[0], voice2[1] / 100

@app.on_event("startup")
def load_models():
    global model, vocoder
    print(f"[🍵] Loading model from {CHECKPOINT_PATH}")
    model = load_matcha("custom_model", CHECKPOINT_PATH, DEVICE)
    model.decoder.solver = ODE_SOLVER

    vocoder = load_vocoder("vocos", DEVICE)
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
    print(f"[🍵] Request: {request.model_dump()}")
    
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = request.input.strip()
    length_scale = request.speed

    if '+' in request.voice:
        id1, weight1, id2, weight2 = parse_voice_mix(request.voice)
        language = VOICES[id1]["lang"]
        voice_mix = [(id1, weight1), (id2, weight2)]
        spk = 0
    else:
        voice_id = int(request.voice)
        language = VOICES[voice_id]["lang"]
        voice_mix = None
        spk = voice_id

    text_processed = process_text(text, language, DEVICE)

    start_t = dt.datetime.now()
    output = model.synthesise(
        text_processed["x"],
        text_processed["x_lengths"],
        n_timesteps=request.steps,
        spks=spk,
        voice_mix=voice_mix,
        length_scale=length_scale,
    )
        
    waveform = to_waveform(output["mel"], vocoder)
    waveform = post_process(waveform)

    t = (dt.datetime.now() - start_t).total_seconds()
    rtf = t * SAMPLE_RATE / waveform.shape[-1]
    print(f"[🍵] Inference time: {t:.2f}s, RTF: {rtf:.4f}")

    mp3_data = convert_to_mp3(waveform)
    return Response(content=mp3_data, media_type="audio/mpeg")


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


# Run it with python -m matcha.server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("matcha.server:app", host="0.0.0.0", port=8000, reload=True)
