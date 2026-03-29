# ── Builder stage: install dependencies ──────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app
COPY pyproject.toml .
COPY server/ server/
COPY shared/ shared/
COPY models.py client.py inference.py ./
COPY scripts/ scripts/
COPY static/ static/
COPY openenv.yaml ./

RUN pip install --no-cache-dir -e .

# ── Final stage: copy only what's needed ─────────────────────
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m appuser

WORKDIR /app

# Copy installed packages and app code from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
