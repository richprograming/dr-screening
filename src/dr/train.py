"""Treino. Uso:
python -m src.dr.train --data data/aptos --epochs 15
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score
from torch.utils.data import DataLoader

from .data import APTOSDataset, class_weights, make_splits
from .model import build_model


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
    qwk = cohen_kappa_score(targets, preds, weights="quadratic")
    return total / len(loader.dataset), qwk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="pasta com train.csv e train_images/")
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--out", type=Path, default=Path("models/best.pt"))
    a = ap.parse_args()

    torch.manual_seed(42)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tr_df, va_df = make_splits(a.data / "train.csv")
    img_dir = a.data / "train_images"
    tr = DataLoader(APTOSDataset(tr_df, img_dir, True, a.size), a.bs, shuffle=True, num_workers=4)
    va = DataLoader(APTOSDataset(va_df, img_dir, False, a.size), a.bs, num_workers=4)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights(tr_df).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, a.epochs)

    best = -1.0
    for ep in range(1, a.epochs + 1):
        tl, tk = run_epoch(model, tr, device, criterion, optimizer)
        vl, vk = run_epoch(model, va, device, criterion)
        sched.step()
        print(f"ep {ep:02d} | train loss {tl:.3f} qwk {tk:.3f} | val loss {vl:.3f} qwk {vk:.3f}")
        if vk > best:
            best = vk
            a.out.parent.mkdir(exist_ok=True)
            torch.save(model.state_dict(), a.out)
    print(f"melhor QWK de validação: {best:.4f}")


if __name__ == "__main__":
    main()
