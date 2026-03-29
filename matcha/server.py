import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

# Set HuggingFace cache BEFORE any imports that might use it
cache_base = Path(os.environ.get("MATCHA_CACHE_DIR", Path.cwd() / ".cache"))
os.environ["HF_HOME"] = str(cache_base / "huggingface")

from parse import parse
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel

import logging
logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
logging.getLogger("torch._inductor").setLevel(logging.ERROR)
logging.getLogger("torch.utils._sympy.interp").setLevel(logging.ERROR)

import torch
torch._inductor.config.fx_graph_cache = True

from matcha.inference import load_matcha, load_vocoder, pipeline, convert_to_mp3, convert_to_opus_ogg, SAMPLE_RATE, ODE_SOLVER, VOICES

CHECKPOINT_PATH = "logs/train/v12/runs/2026-03-27_21-07-50/checkpoints/checkpoint_epoch=144.ckpt"
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", CHECKPOINT_PATH)
model = None
vocoder = None

MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", 2000))

# length_scale is inverse of client speed: higher = slower speech
LENGTH_SCALE_MIN      = 0.1   # fastest (client speed=2.0)
LENGTH_SCALE_MAX      = 2.0   # slowest (client speed=0.1)

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vocoder
    print(f"[🍵] Loading model from {CHECKPOINT_PATH}...")
    model = load_matcha("custom_model", CHECKPOINT_PATH)
    model.decoder.solver = ODE_SOLVER
    vocoder = load_vocoder("vocos")
    print("[🍵] Compiling the model...")
    model.decoder.estimator = torch.compile(model.decoder.estimator, mode="reduce-overhead", dynamic=True)
    for _ in range(3):
        pipeline(model, vocoder, "Warming up.", "en-us")
    print("[🍵] Model loaded.")
    yield


app = FastAPI(title="Matcha-TTS Inference Server", lifespan=lifespan)

class InferenceRequest(BaseModel):
    input: str
    voice: int | str = 0
    response_format: str = "mp3"
    speed: float = 1.0
    steps: int = 15


def parse_voice_mix(voice_str: str):
    """Parse voice mix string like '2(70)+6(30)' into (id1, weight1, id2, weight2)"""
    parts = voice_str.split('+')
    voice1 = parse("{:d}({:d})", parts[0])
    voice2 = parse("{:d}({:d})", parts[1])
    return voice1[0], voice1[1] / 100, voice2[0], voice2[1] / 100


@app.get("/")
def root():
    return {"status": "ok", "message": "Matcha-TTS server is running"}

@app.get("/api/v1/speak")
@app.get("/prod/speak/evie")
@app.get("/test/speak/evie")
def get_voices():
    return VOICES

@app.post("/v1/audio/speech")
@app.post("/api/v1/speak")
@app.post("/prod/speak/evie")
@app.post("/test/speak/evie")
async def speak(request: InferenceRequest):
    if len(request.input) > MAX_TEXT_LENGTH:
        print(f"[🍵] ERROR: Text exceeds {MAX_TEXT_LENGTH} characters.")
        raise HTTPException(status_code=400, detail=f"Text exceeds {MAX_TEXT_LENGTH} characters")

    print(f"[🍵] Request: {request.model_dump()}")

    if '+' in str(request.voice):
        id1, weight1, id2, weight2 = parse_voice_mix(request.voice)
        language = VOICES[id1]["lang"]
        voice_mix = [(id1, weight1), (id2, weight2)]
        speaker = 0
    else:
        voice_id = int(request.voice)
        language = VOICES[voice_id]["lang"]
        voice_mix = None
        speaker = voice_id

    t = time.perf_counter()
    if voice_mix is not None:
        scale_correction = sum(VOICES[spk_id]["scale_correction"] * weight for spk_id, weight in voice_mix)
    else:
        scale_correction = VOICES[speaker]["scale_correction"]
    length_scale = max(LENGTH_SCALE_MIN, min(LENGTH_SCALE_MAX, 1.0 / request.speed))
    waveform = pipeline(model, vocoder, request.input.strip(), language, speaker, voice_mix, request.steps, scale_correction, length_scale)
    elapsed = time.perf_counter() - t
    audio_duration = waveform.shape[-1] / SAMPLE_RATE
    print(f"[🍵] Total time: {elapsed:.2f}s | RTF: {elapsed / audio_duration:.4f}")

    headers = {
        "Content-Disposition": "attachment; filename=speech.mp3",
        "Cache-Control": "no-cache",
    }
    if request.response_format == "mp3":
        return Response(content=convert_to_mp3(waveform), media_type="audio/mpeg", headers=headers)
    return Response(content=convert_to_opus_ogg(waveform), media_type="audio/ogg", headers=headers)


@app.get("/health")
async def health_check():
    if model is None:
        print(f"[🍵] INFO: Model not loaded yet.")
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "healthy"}


# Run it with python -m matcha.server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("matcha.server:app", host="0.0.0.0", port=8000, reload=False)
