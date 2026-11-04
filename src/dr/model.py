"""EfficientNet-B0 com transfer learning para classificação de 5 graus de RD."""
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    m = models.efficientnet_b0(weights=weights)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes))
    return m


def load_model(weights: Path | str, device: str = "cpu") -> nn.Module:
    """Carrega um state_dict salvo por train.py, pronto para inferência."""
    model = build_model(pretrained=False)
    # weights_only=True impede a execução de código embutido em um .pt de terceiros.
    model.load_state_dict(torch.load(weights, map_location=device, weights_only=True))
    return model.to(device).eval()
