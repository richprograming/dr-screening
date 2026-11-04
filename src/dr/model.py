"""EfficientNet-B0 com transfer learning para classificação de 5 graus de RD."""
import torch.nn as nn
from torchvision import models


def build_model(num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
    m = models.efficientnet_b0(weights=weights)
    in_f = m.classifier[1].in_features
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_f, num_classes))
    return m
