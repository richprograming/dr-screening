"""Demo em Gradio, pronta para o Hugging Face Spaces (veja docs/huggingface_space.md).

Pesos: usa DR_WEIGHTS (padrão models/best.pt). Se o arquivo não existir e DR_WEIGHTS_REPO
estiver definido (ex.: usuario/dr-screening), baixa best.pt desse repositório do Hugging Face.
"""
import os
from pathlib import Path

import gradio as gr

from src.dr.inference import DISCLAIMER, Predictor

_predictor: Predictor | None = None


def get_predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        weights = Path(os.getenv("DR_WEIGHTS", "models/best.pt"))
        repo = os.getenv("DR_WEIGHTS_REPO")
        if not weights.exists() and repo:
            from huggingface_hub import hf_hub_download

            weights = hf_hub_download(repo_id=repo, filename="best.pt")
        elif not weights.exists():
            raise gr.Error(f"Pesos não encontrados em {weights}. Treine o modelo ou defina DR_WEIGHTS_REPO.")
        _predictor = Predictor(weights)
    return _predictor


def classify(image):
    if image is None:
        raise gr.Error("Envie uma imagem de fundo de olho.")
    pred = get_predictor()(image)
    referral = "sim" if pred.refer else "não"
    summary = (
        f"**Grau mais provável:** {pred.grade} ({pred.label})  \n"
        f"**Encaminhamento (grau ≥ 2):** {referral}, probabilidade estimada de {pred.refer_probability:.0%}"
    )
    return pred.probabilities, pred.processed, pred.gradcam, summary


demo = gr.Interface(
    fn=classify,
    inputs=gr.Image(type="numpy", image_mode="RGB", label="Imagem de fundo de olho"),
    outputs=[
        gr.Label(label="Probabilidades por grau"),
        gr.Image(label="Pré-processada"),
        gr.Image(label="Grad-CAM"),
        gr.Markdown(),
    ],
    title="Triagem de Retinopatia Diabética",
    description=f"⚠️ {DISCLAIMER} Veja o Model Card no repositório.",
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
