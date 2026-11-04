"""API FastAPI: recebe imagem de fundo de olho e devolve grau, probabilidades e Grad-CAM."""
import base64
import os

import cv2
import numpy as np
import torch
from fastapi import FastAPI, File, HTTPException, UploadFile

from src.dr.data import CLASSES, get_transforms, preprocess
from src.dr.gradcam import GradCAM, overlay
from src.dr.model import build_model

SIZE = 320
WEIGHTS = os.getenv("DR_WEIGHTS", "models/best.pt")

app = FastAPI(title="Triagem de Retinopatia Diabética", version="0.1.0")
model = build_model(pretrained=False)
model.load_state_dict(torch.load(WEIGHTS, map_location="cpu"))
model.eval()
cam_engine = GradCAM(model, model.features[-1])
tf = get_transforms(train=False, size=SIZE)

DISCLAIMER = "Projeto educacional. Não é dispositivo médico e não deve ser usado para diagnóstico."


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = np.frombuffer(await file.read(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "Imagem inválida")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    proc = preprocess(img, SIZE)
    x = tf(proc).unsqueeze(0).requires_grad_(True)

    cam, idx, probs = cam_engine(x)
    vis = overlay(proc, cam)
    ok, buf = cv2.imencode(".png", cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    return {
        "grade": idx,
        "label": CLASSES[idx],
        "refer": idx >= 2,
        "probabilities": {c: round(float(p), 4) for c, p in zip(CLASSES, probs)},
        "gradcam_png_base64": base64.b64encode(buf).decode(),
        "disclaimer": DISCLAIMER,
    }
