"""Procura imagens duplicadas entre treino e validação e mede o impacto nas métricas.

Compara imagens por hash perceptual (pHash) e reavalia o modelo sem os casos de validação
que têm cópia no treino. Uso:
python scripts/check_duplicates.py --data data/aptos --weights models/best.pt

Grava em docs/ um relatório só com contagens e métricas, que pode ser publicado. A lista de
pares fica em duplicate_pairs.csv, dentro de --data, pois cita arquivos do dataset.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # permite rodar como script

import cv2
import numpy as np
import pandas as pd
import torch

from src.dr.data import APTOSDataset, crop_black_borders, load_rgb, make_loader, make_splits
from src.dr.evaluate import compute_metrics
from src.dr.model import load_model

THRESHOLDS = (0, 2, 4, 6, 8)  # bits diferentes, de 64; quanto maior, mais falsos positivos
CORRELATIONS = (0.6, 0.8, 0.9, 0.95)  # correlação dos detalhes finos entre duas imagens


def phash(img_rgb: np.ndarray) -> np.ndarray:
    """Hash de 64 bits, tolerante a mudanças de resolução, brilho e recorte das bordas."""
    gray = cv2.cvtColor(crop_black_borders(img_rgb), cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].flatten()
    return low > np.median(low[1:])  # ignora o termo DC, que só reflete o brilho médio


def cross_split_distances(val_hashes: np.ndarray, train_hashes: np.ndarray) -> np.ndarray:
    """Matriz de distâncias de Hamming: uma linha por imagem de validação."""
    return np.stack([(h != train_hashes).sum(1) for h in val_hashes])


def vessel_signature(img_rgb: np.ndarray, size: int = 96) -> np.ndarray:
    """Vetor unitário com os detalhes finos da retina. O filtro passa-alta e a máscara removem
    a iluminação e o contorno do disco, que fazem olhos diferentes parecerem iguais."""
    gray = cv2.cvtColor(crop_black_borders(img_rgb), cv2.COLOR_RGB2GRAY)
    small = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    detail = small - cv2.GaussianBlur(small, (0, 0), size / 30)
    yy, xx = np.ogrid[:size, :size]
    center = (size - 1) / 2
    inner = (yy - center) ** 2 + (xx - center) ** 2 <= (0.85 * size / 2) ** 2
    values = detail[inner]
    values -= values.mean()
    return values / (np.linalg.norm(values) + 1e-8)


def describe_all(df: pd.DataFrame, img_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    hashes, signatures = [], []
    for i, id_code in enumerate(df.id_code, 1):
        img = load_rgb(img_dir / f"{id_code}.png")
        hashes.append(phash(img))
        signatures.append(vessel_signature(img))
        if i % 500 == 0:
            print(f"  {i}/{len(df)} imagens", flush=True)
    return np.stack(hashes), np.stack(signatures)


def predict_probs(df: pd.DataFrame, img_dir: Path, weights: Path, size: int, workers) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = make_loader(APTOSDataset(df, img_dir, False, size), 32, workers=workers)
    model = load_model(weights, device)
    with torch.no_grad():
        return np.concatenate([model(x.to(device)).softmax(1).cpu().numpy() for x, _ in loader])


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--out", type=Path, default=Path("docs/duplicates_report.json"))
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)

    img_dir = a.data / "train_images"
    tr_df, va_df = make_splits(a.data / "train.csv", limit=a.limit)
    tr_df, va_df = tr_df.reset_index(drop=True), va_df.reset_index(drop=True)
    print(f"calculando hashes de {len(tr_df) + len(va_df)} imagens")
    va_hashes, va_signatures = describe_all(va_df, img_dir)
    tr_hashes, tr_signatures = describe_all(tr_df, img_dir)
    dist = cross_split_distances(va_hashes, tr_hashes)
    nearest = dist.min(1)
    corr = va_signatures @ tr_signatures.T  # correlação entre todos os pares, sem depender do hash
    best_corr = corr.max(1)

    print("reavaliando o modelo na validação")
    ys = va_df.diagnosis.to_numpy()
    probs = predict_probs(va_df, img_dir, a.weights, a.size, a.workers)

    report = {"n_train": len(tr_df), "n_val": len(va_df), "full_validation": compute_metrics(ys, probs), "by_threshold": []}
    for t in THRESHOLDS:
        v, j = np.nonzero(dist <= t)
        keep = nearest > t
        conflicts = int((va_df.diagnosis.to_numpy()[v] != tr_df.diagnosis.to_numpy()[j]).sum())
        metrics = compute_metrics(ys[keep], probs[keep])
        row = {
            "max_bit_distance": t,
            "cross_split_pairs": len(v),
            "val_images_with_match": int((~keep).sum()),
            "pairs_with_different_labels": conflicts,
            "metrics_without_matches": {k: metrics[k] for k in ("n", "qwk", "auc_referral", "accuracy", "recall_per_class")},
        }
        report["by_threshold"].append(row)
        print(
            f"distância <= {t}: {row['val_images_with_match']} de {len(va_df)} na validação têm par no treino, "
            f"{conflicts} pares com rótulos diferentes | sem eles: QWK {metrics['qwk']:.4f}, AUC {metrics['auc_referral']}"
        )

    edges = [-1.0, 0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1.01]
    counts, _ = np.histogram(best_corr, bins=edges)
    report["best_correlation_histogram"] = {f"{lo:g} a {hi:g}": int(c) for lo, hi, c in zip(edges, edges[1:], counts)}
    print("maior correlação de cada imagem de validação com o treino:", report["best_correlation_histogram"])
    report["by_correlation"] = []
    for t in CORRELATIONS:
        v, j = np.nonzero(corr >= t)
        keep = best_corr < t
        conflicts = int((va_df.diagnosis.to_numpy()[v] != tr_df.diagnosis.to_numpy()[j]).sum())
        metrics = compute_metrics(ys[keep], probs[keep])
        report["by_correlation"].append(
            {
                "min_correlation": t,
                "cross_split_pairs": len(v),
                "val_images_with_match": int((~keep).sum()),
                "pairs_with_different_labels": conflicts,
                "metrics_without_matches": {k: metrics[k] for k in ("n", "qwk", "auc_referral", "accuracy", "recall_per_class")},
            }
        )
        print(
            f"correlação >= {t}: {int((~keep).sum())} de {len(va_df)} têm par no treino, {conflicts} pares com rótulos "
            f"diferentes | sem eles: QWK {metrics['qwk']:.4f}, AUC {metrics['auc_referral']:.4f}, recall {metrics['recall_per_class']}"
        )

    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    v, j = np.nonzero((dist <= max(THRESHOLDS)) | (corr >= min(CORRELATIONS)))
    pd.DataFrame(
        {
            "val_id": va_df.id_code.to_numpy()[v],
            "train_id": tr_df.id_code.to_numpy()[j],
            "bit_distance": dist[v, j],
            "correlation": corr[v, j],
            "val_label": va_df.diagnosis.to_numpy()[v],
            "train_label": tr_df.diagnosis.to_numpy()[j],
        }
    ).sort_values("correlation", ascending=False).to_csv(a.data / "duplicate_pairs.csv", index=False)
    print(f"relatório: {a.out} | pares: {a.data / 'duplicate_pairs.csv'}")


if __name__ == "__main__":
    main()
