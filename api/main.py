"""API FastAPI: recebe imagem de fundo de olho e devolve grau, probabilidades e Grad-CAM."""
import base64
import logging
import os
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.dr.inference import DISCLAIMER, Predictor

SIZE = 320
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
log = logging.getLogger("uvicorn.error")
predictor: Predictor | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Sem pesos a API sobe mesmo assim: /health informa e /predict responde 503."""
    global predictor
    weights = os.getenv("DR_WEIGHTS", "models/best.pt")
    if os.path.exists(weights):
        predictor = Predictor(weights, SIZE)
    else:
        log.warning("Pesos não encontrados em %s; /predict responderá 503.", weights)
    yield
    predictor = None


app = FastAPI(title="Triagem de Retinopatia Diabética", version="0.1.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict")
def predict(file: UploadFile = File(...)):  # def: o FastAPI roda em thread, sem travar o event loop
    if predictor is None:
        raise HTTPException(503, "Modelo não carregado: treine o modelo ou defina DR_WEIGHTS")
    raw = file.file.read(MAX_UPLOAD_BYTES + 1)
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "Imagem maior que 20 MB")
    # cv2.imdecode levanta exceção com buffer vazio, em vez de devolver None.
    img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR) if raw else None
    if img is None:
        raise HTTPException(400, "Imagem inválida")

    pred = predictor(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    ok, buf = cv2.imencode(".png", cv2.cvtColor(pred.gradcam, cv2.COLOR_RGB2BGR))
    if not ok:
        raise HTTPException(500, "Falha ao gerar o Grad-CAM")
    return {
        "grade": pred.grade,
        "label": pred.label,
        "refer": pred.refer,
        "refer_probability": round(pred.refer_probability, 4),
        "probabilities": {c: round(p, 4) for c, p in pred.probabilities.items()},
        "gradcam_png_base64": base64.b64encode(buf).decode(),
        "disclaimer": DISCLAIMER,
    }
