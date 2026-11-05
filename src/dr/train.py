"""Treino. Uso:
python -m src.dr.train --data data/aptos --epochs 15

Teste rápido (subconjunto pequeno, 1 época):
python -m src.dr.train --data data/aptos --epochs 1 --limit 100 --out models/smoke.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score

from .data import CLASSES, APTOSDataset, class_weights, make_loader, make_splits
from .model import build_model


def qwk(targets, preds) -> float:
    # labels fixos: sem eles, uma classe ausente aproxima os graus vizinhos e distorce os pesos.
    return cohen_kappa_score(targets, preds, labels=list(range(len(CLASSES))), weights="quadratic")


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    preds, targets, total = [], [], 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * len(y)
            preds += out.argmax(1).cpu().tolist()
            targets += y.cpu().tolist()
    return total / len(targets), qwk(targets, preds)  # len(targets): o loader pode descartar um lote


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="pasta com train.csv e train_images/")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--out", type=Path, default=Path("models/best.pt"))
    ap.add_argument("--workers", type=int, default=None, help="padrão: 0 no Windows/macOS, até 4 no Linux")
    ap.add_argument("--limit", type=int, default=None, help="usa só N imagens (teste rápido, não para métricas)")
    ap.add_argument("--no-pretrained", action="store_true", help="não baixa os pesos ImageNet")
    a = ap.parse_args(argv)

    torch.manual_seed(42)
    np.random.seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"dispositivo: {device}")
    tr_df, va_df = make_splits(a.data / "train.csv", limit=a.limit)
    img_dir = a.data / "train_images"
    tr = make_loader(APTOSDataset(tr_df, img_dir, True, a.size), a.bs, train=True, workers=a.workers)
    va = make_loader(APTOSDataset(va_df, img_dir, False, a.size), a.bs, workers=a.workers)

    model = build_model(pretrained=not a.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(tr_df).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, a.epochs)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for ep in range(1, a.epochs + 1):
        tl, tk = run_epoch(model, tr, device, criterion, optimizer)
        vl, vk = run_epoch(model, va, device, criterion)
        sched.step()
        print(f"ep {ep:02d} | train loss {tl:.3f} qwk {tk:.3f} | val loss {vl:.3f} qwk {vk:.3f}", flush=True)
        if ep == 1 or vk > best:  # ep == 1: sempre salva, mesmo se o QWK vier NaN
            best = vk
            torch.save(model.state_dict(), a.out)
    print(f"melhor QWK de validação: {best:.4f}")


if __name__ == "__main__":
    main()
