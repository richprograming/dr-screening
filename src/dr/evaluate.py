"""Avaliação detalhada: QWK, relatório por classe, matriz de confusão e AUC
de encaminhamento (RD moderada+ vs. resto). Uso:
python -m src.dr.evaluate --data data/aptos --weights models/best.pt

Grava metrics.json e confusion_matrix.png em --out-dir (padrão: docs/).
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # só salva arquivos; funciona em servidor, CI e Colab
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from .data import CLASSES, APTOSDataset, make_loader, make_splits
from .model import load_model
from .train import qwk


def compute_metrics(ys: np.ndarray, probs: np.ndarray) -> dict:
    labels = list(range(len(CLASSES)))  # fixos: um subconjunto pode não ter todas as classes
    preds = probs.argmax(1)
    report = classification_report(
        ys, preds, labels=labels, target_names=CLASSES, zero_division=0, output_dict=True
    )
    refer = (ys >= 2).astype(int)  # RD moderada ou pior = encaminhar
    # A AUC só existe se houver casos dos dois lados.
    auc = float(roc_auc_score(refer, probs[:, 2:].sum(1))) if len(set(refer)) == 2 else None
    return {
        "n": int(len(ys)),
        "qwk": float(qwk(ys, preds)),
        "auc_referral": auc,
        "accuracy": float((preds == ys).mean()),
        "recall_per_class": {c: float(report[c]["recall"]) for c in CLASSES},
        "support_per_class": {c: int(report[c]["support"]) for c in CLASSES},
        "confusion_matrix": confusion_matrix(ys, preds, labels=labels).tolist(),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--out-dir", type=Path, default=Path("docs"))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None, help="igual ao usado no treino, se houver")
    a = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    _, va_df = make_splits(a.data / "train.csv", limit=a.limit)
    loader = make_loader(APTOSDataset(va_df, a.data / "train_images", False, a.size), 32, workers=a.workers)
    model = load_model(a.weights, device)

    probs, ys = [], []
    with torch.no_grad():
        for x, y in loader:
            probs.append(model(x.to(device)).softmax(1).cpu().numpy())
            ys += y.tolist()
    probs, ys = np.concatenate(probs), np.array(ys)
    preds = probs.argmax(1)
    m = compute_metrics(ys, probs)
    m["limit"] = a.limit  # registra se o resultado veio de um subconjunto
    m["weights"] = a.weights.name

    print(f"QWK: {m['qwk']:.4f}")
    print(
        classification_report(
            ys, preds, labels=list(range(len(CLASSES))), target_names=CLASSES, digits=3, zero_division=0
        )
    )
    auc = "indefinida" if m["auc_referral"] is None else f"{m['auc_referral']:.4f}"
    print(f"AUC (encaminhar, grau>=2): {auc}")

    a.out_dir.mkdir(parents=True, exist_ok=True)
    (a.out_dir / "metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
    ConfusionMatrixDisplay.from_predictions(
        ys, preds, labels=list(range(len(CLASSES))), display_labels=CLASSES, cmap="Blues", xticks_rotation=45
    )
    plt.tight_layout()
    plt.savefig(a.out_dir / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"resultados salvos em {a.out_dir}/ (metrics.json, confusion_matrix.png)")


if __name__ == "__main__":
    main()
