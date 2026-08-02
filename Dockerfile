# syntax=docker/dockerfile:1.7

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/cache/huggingface \
    CW_MODEL=small \
    CW_BACKEND=transformers \
    CW_DEVICE=cpu \
    CW_COMPUTE_TYPE=float32

RUN apt-get update \
    && apt-get install --yes --no-install-recommends ca-certificates libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY crisperwhisper ./crisperwhisper

RUN python -m pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cpu "torch>=2.4" \
    && python -m pip install --no-cache-dir ".[transformers]" \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /data /cache/huggingface \
    && chown -R app:app /data /cache/huggingface

USER app
WORKDIR /data

ENTRYPOINT ["crisperwhisper"]
CMD ["--help"]
