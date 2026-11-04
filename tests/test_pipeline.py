import numpy as np
import torch

from src.dr.data import preprocess
from src.dr.gradcam import GradCAM
from src.dr.model import build_model


def test_preprocess_shape():
    img = np.random.randint(0, 255, (500, 600, 3), dtype=np.uint8)
    assert preprocess(img, 320).shape == (320, 320, 3)


def test_model_output():
    m = build_model(pretrained=False).eval()
    assert m(torch.randn(2, 3, 320, 320)).shape == (2, 5)


def test_gradcam_range():
    m = build_model(pretrained=False)
    cam, idx, probs = GradCAM(m, m.features[-1])(torch.randn(1, 3, 320, 320).requires_grad_(True))
    assert cam.shape == (320, 320)
    assert 0.0 <= cam.min() and cam.max() <= 1.0
    assert abs(probs.sum() - 1) < 1e-4
