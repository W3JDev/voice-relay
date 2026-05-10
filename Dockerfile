FROM python:3.12-slim

LABEL maintainer="OpenCLAW Voice Relay"
LABEL description="Phase 1 Voice Relay - lightweight CPU-only relay mode"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY src/ src/

# Relay-mode defaults (no GPU, API-first)
ENV OPENCLAW_HOST=0.0.0.0
ENV OPENCLAW_PORT=8765
ENV OPENCLAW_STT_DEVICE=cpu
ENV OPENCLAW_REQUIRE_AUTH=false

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${OPENCLAW_PORT:-8765}/health || exit 1

CMD uvicorn src.server.main:app --host 0.0.0.0 --port ${OPENCLAW_PORT:-8765}
