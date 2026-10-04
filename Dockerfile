# Support-ticket resolution assistant: one stateless container serving the API and the UI.
# State (patterns, resolutions, FAISS index) is NOT baked in: mount it at /data/state (or set TICKETRAG_STATE).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/hf

WORKDIR /app

# CPU-only torch first: avoids the multi-GB CUDA wheel and keeps the image small.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

# Dependencies in their own layer so code changes do not re-download them.
COPY requirements.txt .
RUN pip install -r requirements.txt

# Bake the embedding model into the image: no network access or cold-start download at runtime.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-en-v1.5')"

COPY pyproject.toml .
COPY src ./src
RUN pip install --no-deps .

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /data/state \
    && chown -R app /data /opt/hf
USER app

# The response cache lives in /tmp (writable even with a read-only root filesystem; lost on restart, which is fine).
ENV TICKETRAG_STATE=/data/state \
    TICKETRAG_CACHE_DIR=/tmp/llm-cache \
    HF_HUB_OFFLINE=1

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"

# One worker per container: every worker would hold its own copy of the index. Scale with replicas instead.
CMD ["uvicorn", "ticketrag.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
