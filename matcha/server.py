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

from matcha.inference import load_matcha, load_vocoder, synthesise, convert_to_mp3, convert_to_opus_ogg, SAMPLE_RATE, ODE_SOLVER

CHECKPOINT_PATH = "logs/train/corpus-small-24k/v1/checkpoint_epoch=579.ckpt"
CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", CHECKPOINT_PATH)
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

MAX_TEXT_LENGTH = int(os.environ.get("MAX_TEXT_LENGTH", 500))


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, vocoder
    print(f"[🍵] Loading model from {CHECKPOINT_PATH}")
    model = load_matcha("custom_model", CHECKPOINT_PATH)
    model.decoder.solver = ODE_SOLVER
    vocoder = load_vocoder("vocos")
    print("[🍵] Models loaded successfully")
    yield


app = FastAPI(title="Matcha-TTS Inference Server", lifespan=lifespan)

class InferenceRequest(BaseModel):
    input: str
    voice: int | str = 0
    response_format: str = "ogg"
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
def speak(request: InferenceRequest):
    if len(request.input) > MAX_TEXT_LENGTH:
        raise HTTPException(status_code=400, detail=f"Text exceeds {MAX_TEXT_LENGTH} characters")

    print(f"[🍵] Request: {request.model_dump()}")

    if '+' in str(request.voice):
        id1, weight1, id2, weight2 = parse_voice_mix(request.voice)
        language = VOICES[id1]["lang"]
        voice_mix = [(id1, weight1), (id2, weight2)]
        spk = 0
    else:
        voice_id = int(request.voice)
        language = VOICES[voice_id]["lang"]
        voice_mix = None
        spk = voice_id

    t = time.perf_counter()
    waveform = synthesise(model, vocoder, request.input.strip(), language, spk, voice_mix, request.steps, request.speed)
    elapsed = time.perf_counter() - t
    audio_duration = waveform.shape[-1] / SAMPLE_RATE
    print(f"[🍵] Total time: {elapsed:.2f}s | RTF: {elapsed / audio_duration:.4f}")

    if request.response_format == "mp3":
        return Response(content=convert_to_mp3(waveform), media_type="audio/mpeg")
    return Response(content=convert_to_opus_ogg(waveform), media_type="audio/ogg")


@app.get("/health")
async def health_check():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "healthy"}


# Run it with python -m matcha.server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("matcha.server:app", host="0.0.0.0", port=8880, reload=False, workers=3)
