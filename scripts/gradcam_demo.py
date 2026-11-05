"""Salva exemplos lado a lado (original, pré-processada, Grad-CAM) em docs/.
Usa imagens da validação, uma por grau por padrão. Uso:
python scripts/gradcam_demo.py --data data/aptos --weights models/best.pt
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # permite rodar como script

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dr.data import CLASSES, load_rgb, make_splits
from src.dr.inference import DISCLAIMER, Predictor


def save_panel(img, pred, true_grade: int, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.2))
    panels = [(img, "Original"), (pred.processed, "Pré-processada"), (pred.gradcam, "Grad-CAM")]
    for ax, (panel, title) in zip(axes, panels):
        ax.imshow(panel)
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle(
        f"Real: {CLASSES[true_grade]} | Previsto: {pred.label} ({pred.probabilities[pred.label]:.0%})"
    )
    fig.text(0.5, 0.02, DISCLAIMER, ha="center", fontsize=8, color="dimgray")
    fig.tight_layout(rect=(0, 0.05, 1, 0.97))
    fig.savefig(out, dpi=120)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("docs"))
    ap.add_argument("--per-class", type=int, default=1)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--limit", type=int, default=None, help="igual ao usado no treino, se houver")
    a = ap.parse_args(argv)

    a.out.mkdir(parents=True, exist_ok=True)
    predictor = Predictor(a.weights, a.size)
    _, va_df = make_splits(a.data / "train.csv", limit=a.limit)
    for grade, group in va_df.groupby("diagnosis"):
        for i, row in enumerate(group.head(a.per_class).itertuples()):
            img = load_rgb(a.data / "train_images" / f"{row.id_code}.png")
            out = a.out / f"gradcam_grau{grade}_{i}.png"
            save_panel(img, predictor(img), int(grade), out)
            print(f"salvo: {out}")


if __name__ == "__main__":
    main()
