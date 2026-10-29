# ================= Stage 1: trainer =================
FROM python:3.10-slim AS trainer
WORKDIR /build
ENV PYTHONPATH=/build PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-train.txt .
RUN pip install --no-cache-dir -r requirements-train.txt

# окремий шар: завантаження датасету (не повторюється, якщо код навчання змінився)
COPY src/__init__.py src/data.py ./src/
RUN python -c "from src.data import get_root; get_root('/build/data')" \
    && rm -f /build/data/*.tar.gz

# навчання й оцінка -> /build/artifacts
COPY src/ ./src/
ARG EPOCHS=10
RUN python -m src.train --classes yes no up down --epochs ${EPOCHS} --seed 42 \
        --data-dir /build/data --out /build/artifacts


# ================= Stage 2: runtime =================
FROM python:3.10-slim AS runtime
WORKDIR /app
ENV PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src/ ./src/
COPY --from=trainer /build/artifacts ./artifacts

ENV MODEL_PATH=/app/artifacts/model.pt \
    PORT=8000 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home appuser
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import os,urllib.request as u; u.urlopen('http://localhost:%s/health' % os.getenv('PORT','8000'))"

CMD ["sh", "-c", "uvicorn src.api:app --host 0.0.0.0 --port ${PORT}"]