"""Gera imagens FALSAS no formato do APTOS, só para testar o pipeline sem baixar o dataset.
Métricas obtidas com estes dados não dizem nada sobre retinopatia. Uso:
python scripts/make_synthetic_data.py --out data/synthetic --per-class 20
"""
import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


def make_image(grade: int, rng: np.random.Generator, size: int = 256) -> np.ndarray:
    """Disco alaranjado sobre fundo preto, com mais manchas quanto maior o grau (BGR)."""
    img = np.zeros((size, size, 3), np.uint8)
    color = [int(c) for c in rng.integers([20, 60, 130], [50, 110, 200])]
    cv2.circle(img, (size // 2, size // 2), size // 2 - 8, color, -1)
    for _ in range(grade * 6):
        r = np.sqrt(rng.random()) * (size // 2 - 24)
        t = rng.random() * 2 * np.pi
        center = (int(size / 2 + r * np.cos(t)), int(size / 2 + r * np.sin(t)))
        cv2.circle(img, center, int(rng.integers(2, 6)), (20, 20, 90), -1)
    return img


def generate(out: Path, per_class: int = 20, size: int = 256, seed: int = 42) -> Path:
    rng = np.random.default_rng(seed)
    (out / "train_images").mkdir(parents=True, exist_ok=True)
    rows = []
    for grade in range(5):
        for i in range(per_class):
            id_code = f"sint_{grade}_{i:04d}"
            cv2.imwrite(str(out / "train_images" / f"{id_code}.png"), make_image(grade, rng, size))
            rows.append({"id_code": id_code, "diagnosis": grade})
    pd.DataFrame(rows).to_csv(out / "train.csv", index=False)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--per-class", type=int, default=20)
    ap.add_argument("--size", type=int, default=256)
    a = ap.parse_args()
    generate(a.out, a.per_class, a.size)
    print(f"{a.per_class * 5} imagens sintéticas em {a.out}")
