"""Avaliação detalhada: QWK, relatório por classe, matriz de confusão e AUC
de encaminhamento (RD moderada+ vs. resto). Uso:
python -m src.dr.evaluate --data data/aptos --weights models/best.pt
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    cohen_kappa_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader

from .data import CLASSES, APTOSDataset, make_splits
from .model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    ap.add_argument("--size", type=int, default=320)
    a = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, va_df = make_splits(a.data / "train.csv")
    loader = DataLoader(APTOSDataset(va_df, a.data / "train_images", False, a.size), 32, num_workers=4)
    model = build_model(pretrained=False).to(device).eval()
    model.load_state_dict(torch.load(a.weights, map_location=device))

    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(model(x.to(device)).softmax(1).cpu().numpy())
            ys += y.tolist()
    probs, ys = np.concatenate(probs), np.array(ys)
    preds = probs.argmax(1)

    print(f"QWK: {cohen_kappa_score(ys, preds, weights='quadratic'):.4f}")
    print(classification_report(ys, preds, target_names=CLASSES, digits=3))
    refer = (ys >= 2).astype(int)  # RD moderada ou pior = encaminhar
    print(f"AUC (encaminhar, grau>=2): {roc_auc_score(refer, probs[:, 2:].sum(1)):.4f}")

    ConfusionMatrixDisplay.from_predictions(ys, preds, display_labels=CLASSES, cmap="Blues", xticks_rotation=45)
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    print("matriz de confusão salva em confusion_matrix.png")


if __name__ == "__main__":
    main()
