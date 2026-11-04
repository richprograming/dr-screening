"""Exporta o modelo para ONNX (inferência leve). Uso:
python -m src.dr.export_onnx --weights models/best.pt --out models/dr.onnx
"""
import argparse
from pathlib import Path

import torch

from .model import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("models/dr.onnx"))
    ap.add_argument("--size", type=int, default=320)
    a = ap.parse_args()

    model = build_model(pretrained=False).eval()
    model.load_state_dict(torch.load(a.weights, map_location="cpu"))
    dummy = torch.randn(1, 3, a.size, a.size)
    torch.onnx.export(
        model, dummy, a.out, input_names=["image"], output_names=["logits"],
        dynamic_axes={"image": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17,
    )
    print(f"exportado: {a.out}")


if __name__ == "__main__":
    main()
