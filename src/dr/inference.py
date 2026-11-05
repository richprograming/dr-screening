"""Inferência com Grad-CAM, compartilhada pela API, pela demo Gradio e por gradcam_demo."""
import threading
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import CLASSES, get_transforms, preprocess
from .gradcam import GradCAM, overlay
from .model import load_model

DISCLAIMER = "Projeto educacional. Não é dispositivo médico e não deve ser usado para diagnóstico."


@dataclass
class Prediction:
    grade: int
    label: str
    refer: bool  # o grau mais provável é moderada ou pior
    refer_probability: float  # P(grau >= 2), a mesma pontuação usada na AUC de encaminhamento
    probabilities: dict[str, float]
    processed: np.ndarray  # imagem pré-processada, RGB
    gradcam: np.ndarray  # Grad-CAM sobreposto à imagem pré-processada, RGB


class Predictor:
    def __init__(self, weights: Path | str, size: int = 320, device: str = "cpu"):
        self.size, self.device = size, device
        self.model = load_model(weights, device)
        self.cam = GradCAM(self.model, self.model.features[-1])
        self.tf = get_transforms(train=False, size=size)
        self._lock = threading.Lock()  # o GradCAM guarda estado entre forward e backward

    def __call__(self, img_rgb: np.ndarray) -> Prediction:
        proc = preprocess(img_rgb, self.size)
        x = self.tf(proc).unsqueeze(0).to(self.device)
        with self._lock:
            cam, idx, probs = self.cam(x)
        return Prediction(
            grade=idx,
            label=CLASSES[idx],
            refer=idx >= 2,
            refer_probability=float(probs[2:].sum()),
            probabilities={c: float(p) for c, p in zip(CLASSES, probs)},
            processed=proc,
            gradcam=overlay(proc, cam),
        )
