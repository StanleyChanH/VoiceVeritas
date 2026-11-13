# VoiceVeritas API service image.
# Build:  docker build -t voiceveritas .
# Run:    docker run -p 8000:8000 -v $(pwd)/runs/myrun:/models voiceveritas

FROM python:3.11-slim

# ffmpeg for mp3/m4a decoding; git for any URL-based installs if needed
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src
RUN pip install --no-cache-dir ".[api]"

# Bake a model at build time (ARG MODEL_PATH) or mount one at /models at runtime.
ENV VV_MODEL=/models/model.joblib
COPY scripts/serve_entrypoint.py ./serve_entrypoint.py

EXPOSE 8000
CMD ["python", "serve_entrypoint.py"]
