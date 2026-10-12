# syntax=docker/dockerfile:1

# =========================================================
# Backend image — FastAPI RAG API served by uvicorn
# =========================================================
FROM python:3.12-slim

# Keep Python lean and predictable inside the container.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so this layer is cached across code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application package.
COPY app ./app

# The vector store and uploaded PDFs live under /data (a mounted volume),
# so they persist across container restarts and image rebuilds.
ENV DATA_DIR=/data

# Run as a non-root user for security; give it ownership of /app and /data.
RUN mkdir -p /data \
    && adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app /data
USER appuser

VOLUME ["/data"]
EXPOSE 8000

# Container-level liveness probe (compose defines its own as well).
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/api/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
