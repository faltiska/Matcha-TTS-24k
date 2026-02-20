# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:13.0.2-base-ubuntu24.04

RUN apt-get update -y && \
    apt-get install -y git cmake g++ curl espeak-ng espeak-ng-data ffmpeg && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    mkdir -p /usr/share/espeak-ng-data && \
    ln -s /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/ && \
    useradd -m -u 1001 appuser && \
    mkdir -p /app/models && \
    chown -R appuser:appuser /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

USER appuser
WORKDIR /app

COPY --chown=appuser:appuser requirements-inference.txt ./
COPY --chown=appuser:appuser matcha ./matcha
COPY --chown=appuser:appuser logs/train/corpus-small-24k/v1/checkpoint_epoch=579.ckpt ./models/checkpoint.ckpt
COPY --chown=appuser:appuser .cache ./.cache

RUN --mount=type=cache,target=/uv-cache,uid=1001 \
    UV_CACHE_DIR=/uv-cache \
    uv venv --python 3.13 && \
    UV_CACHE_DIR=/uv-cache uv pip install torch torchaudio torchvision torchcodec --index-url https://download.pytorch.org/whl/cu130 && \
    UV_CACHE_DIR=/uv-cache uv pip install -r requirements-inference.txt

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data \
    MATCHA_CACHE_DIR=/app/.cache \
    TORCH_HOME=/app/.cache/torch \
    CHECKPOINT_PATH=/app/models/checkpoint.ckpt \
    MAX_TEXT_LENGTH=500

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "matcha.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
