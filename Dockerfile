FROM python:3.12-slim

WORKDIR /app

# Install system deps (spaCy model needs network on first import; do it here)
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
COPY src/ src/
COPY service/ service/

RUN pip install --no-cache-dir -e . \
    && python -m spacy download en_core_web_sm

# ISAACUS_API_KEY must be injected at runtime:
#   docker run -e ISAACUS_API_KEY=sk-... -p 8000:8000 grounding-api
ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uvicorn", "service.app:app", "--host", "0.0.0.0", "--port", "8000"]
