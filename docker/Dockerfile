FROM --platform=$BUILDPLATFORM nvidia/cuda:12.8.1-base-ubuntu24.04

RUN apt-get update -y && \
    apt-get install -y python3.10 python3-venv espeak-ng espeak-ng-data git libsndfile1 curl && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    mkdir -p /usr/share/espeak-ng-data && \
    ln -s /usr/lib/*/espeak-ng-data/* /usr/share/espeak-ng-data/ && \
    curl -LsSf https://astral.sh/uv/install.sh | sh && \
    mv /root/.local/bin/uv /usr/local/bin/ && \
    useradd -m -u 1001 appuser && \
    chown -R appuser:appuser /app

USER appuser
WORKDIR /app

COPY --chown=appuser:appuser . .

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data \
    MATCHA_CACHE_DIR=/app/.cache \
    TORCH_HOME=/app/.cache/torch

CMD ["python", "-m", "matcha.cli", "--text", "Hello world", "--language", "en-us", "--vocoder", "vocos"]
