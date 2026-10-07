# syntax=docker/dockerfile:1.4
FROM nvidia/cuda:13.1.2-base-ubuntu24.04

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

COPY --chown=appuser:appuser pyproject.toml uv.lock ./
COPY --chown=appuser:appuser .cache ./.cache

# Install only the inference dependencies ([project].dependencies), at the exact versions in uv.lock,
# including the pinned +cu132 torch builds. --no-default-groups skips the `train` group.
# --no-install-project: the code is used straight from /app via PYTHONPATH.
# UV_LINK_MODE=copy: the cache mount is on a different filesystem, so hardlinking is not possible.
RUN --mount=type=cache,target=/uv-cache,uid=1001 \
    UV_CACHE_DIR=/uv-cache UV_LINK_MODE=copy UV_PYTHON=3.13 \
    uv sync --locked --no-default-groups --no-install-project

COPY --chown=appuser:appuser matcha ./matcha
COPY --chown=appuser:appuser docker/checkpoint.ckpt ./models/checkpoint.ckpt

ARG IMAGE_VERSION=unknown
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PHONEMIZER_ESPEAK_PATH=/usr/bin \
    PHONEMIZER_ESPEAK_DATA=/usr/share/espeak-ng-data \
    ESPEAK_DATA_PATH=/usr/share/espeak-ng-data \
    MATCHA_CACHE_DIR=/app/.cache \
    TORCH_HOME=/app/.cache/torch \
    CHECKPOINT_PATH=/app/models/checkpoint.ckpt \
    MAX_TEXT_LENGTH=1000 \
    IMAGE_VERSION=$IMAGE_VERSION

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "matcha.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
