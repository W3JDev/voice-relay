FROM python:3.12-slim

LABEL maintainer="OpenCLAW Voice Relay"
LABEL description="Phase 2 Voice Relay - multi-agent routing, admin API, SQLite persistence"

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (gcc needed for webrtcvad C extension)
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ffmpeg \
        libsndfile1 \
        curl \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 for building the React widget
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y nodejs

WORKDIR /app

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Remove build tools to shrink image (runtime doesn't need gcc)
RUN apt-get purge -y --auto-remove build-essential && rm -rf /var/lib/apt/lists/*

# Copy application source
COPY src/ src/

# Build the React widget
COPY widget/ widget/
RUN cd widget && npm install && npm run build && \
    mkdir -p /app/src/client/widget && \
    cp -r dist/* /app/src/client/widget/

# Relay-mode defaults (no GPU, API-first)
ENV OPENCLAW_HOST=0.0.0.0
ENV OPENCLAW_PORT=8765
ENV OPENCLAW_STT_DEVICE=cpu
ENV OPENCLAW_REQUIRE_AUTH=false

# SQLite persistence
ENV OPENCLAW_DB_PATH=/data/voice_relay.db
VOLUME ["/data"]

EXPOSE 8765

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:${OPENCLAW_PORT:-8765}/health || exit 1

CMD uvicorn src.server.main:app --host 0.0.0.0 --port ${OPENCLAW_PORT:-8765}
