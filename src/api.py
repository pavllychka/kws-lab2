import io
import json
import os
import time

import soundfile as sf
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.data import N_MELS, preprocess
from src.model import SmallCNN

MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.pt")
MODEL_VERSION = os.getenv("MODEL_VERSION", "1.0.0")
ART_DIR = os.path.dirname(MODEL_PATH)

with open(os.path.join(ART_DIR, "labels.json"), encoding="utf-8") as f:
    LABELS = json.load(f)
with open(os.path.join(ART_DIR, "metrics.json"), encoding="utf-8") as f:
    METRICS = json.load(f)

app = FastAPI(title="Speech Commands API")

# модель завантажується один раз при старті
model = SmallCNN(len(LABELS))
model.load_state_dict(
    torch.load(MODEL_PATH, map_location="cpu", weights_only=True))
model.eval()
with torch.no_grad():                      # прогрів
    for _ in range(5):
        model(torch.zeros(1, 1, N_MELS, 101))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    return {
        "architecture": "SmallCNN (3 x Conv-BN-ReLU-MaxPool + GAP + FC)",
        "classes": LABELS,
        "parameters": METRICS["parameters"],
        "model_size_kb": round(os.path.getsize(MODEL_PATH) / 1024, 1),
        "accuracy": METRICS["accuracy"],      # у відсотках
        "model_version": MODEL_VERSION,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    raw = await file.read()
    try:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32")
    except Exception:
        raise HTTPException(400, "Не вдалося прочитати аудіо: потрібен WAV")
    if data.size == 0:
        raise HTTPException(400, "Порожній аудіофайл")
    wave = torch.from_numpy(data)
    wave = wave.unsqueeze(0) if wave.ndim == 1 else wave.T.contiguous()

    t0 = time.perf_counter()               # лише препроцесинг + інференс
    with torch.no_grad():
        x = preprocess(wave, sr).unsqueeze(0)
        probs = torch.softmax(model(x), dim=1)[0]
    latency_ms = (time.perf_counter() - t0) * 1000

    idx = int(probs.argmax())
    return {
        "label": LABELS[idx],
        "confidence": round(float(probs[idx]), 4),
        "probabilities": {c: round(float(p), 4) for c, p in zip(LABELS, probs)},
        "latency_ms": round(latency_ms, 2),
        "model_version": MODEL_VERSION,
    }