"""Exporta o modelo para ONNX (inferência leve). Uso:
python -m src.dr.export_onnx --weights models/best.pt --out models/dr.onnx
"""
import argparse
from pathlib import Path

import torch

from .model import load_model


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", type=Path, default=Path("models/best.pt"))
    ap.add_argument("--out", type=Path, default=Path("models/dr.onnx"))
    ap.add_argument("--size", type=int, default=320)
    a = ap.parse_args(argv)

    model = load_model(a.weights)
    dummy = torch.randn(2, 3, a.size, a.size)  # lote > 1: com 1, o exportador pode fixar o tamanho
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, (dummy,), str(a.out), input_names=["image"], output_names=["logits"],
        dynamic_shapes=({0: torch.export.Dim("batch")},), opset_version=18,
        external_data=False,  # um único arquivo .onnx, sem .onnx.data ao lado
    )
    print(f"exportado: {a.out}")


if __name__ == "__main__":
    main()
