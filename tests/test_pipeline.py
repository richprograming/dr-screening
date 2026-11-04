import numpy as np
import pandas as pd
import pytest
import torch
import torch.nn as nn

from src.dr.data import APTOSDataset, class_weights, get_transforms, load_rgb, preprocess
from src.dr.evaluate import compute_metrics
from src.dr.gradcam import GradCAM
from src.dr.model import build_model
from src.dr.train import qwk


def calibrated_model():
    """Sem pré-treino e em eval, as ativações finais somem (~1e-13) e o Grad-CAM sai vazio.
    Alguns forwards em modo treino ajustam o BatchNorm e deixam a rede em escala realista."""
    torch.manual_seed(0)
    m = build_model(pretrained=False).train()
    with torch.no_grad():
        for _ in range(3):
            m(torch.randn(4, 3, 64, 64))
    return m.eval()


def test_preprocess_shape():
    img = np.random.randint(0, 255, (500, 600, 3), dtype=np.uint8)
    assert preprocess(img, 320).shape == (320, 320, 3)


def test_model_output():
    m = build_model(pretrained=False).eval()
    assert m(torch.randn(2, 3, 320, 320)).shape == (2, 5)


def test_gradcam_range():
    m = calibrated_model()
    cam, idx, probs = GradCAM(m, m.features[-1])(torch.randn(1, 3, 320, 320))
    assert cam.shape == (320, 320)
    assert 0.0 <= cam.min() and cam.max() <= 1.0
    assert cam.max() > 0.5, "mapa vazio: a normalização deveria levar o pico para perto de 1"
    assert abs(probs.sum() - 1) < 1e-4


def test_gradcam_hooks_capture_real_gradients():
    m = calibrated_model()
    x = torch.randn(1, 3, 96, 96)
    store = {}

    def keep(_m, _i, output):
        output.retain_grad()
        store["acts"] = output

    h = m.features[-1].register_forward_hook(keep)
    m.zero_grad()
    m(x)[0, 2].backward()
    h.remove()

    engine = GradCAM(m, m.features[-1])
    engine(x, class_idx=2)
    assert torch.allclose(engine.acts, store["acts"].detach())
    assert torch.allclose(engine.grads, store["acts"].grad)


def test_gradcam_depends_on_target_class():
    m = calibrated_model()
    engine = GradCAM(m, m.features[-1])
    x = torch.randn(1, 3, 96, 96)
    assert np.abs(engine(x, 0)[0] - engine(x, 4)[0]).mean() > 1e-3


def test_gradcam_localizes_known_region():
    """A rede só soma a intensidade por pixel: o mapa precisa apontar para a região clara."""
    net = nn.Sequential(nn.Conv2d(3, 1, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    nn.init.constant_(net[0].weight, 1.0)
    nn.init.constant_(net[0].bias, 0.0)
    x = torch.zeros(1, 3, 32, 32)
    x[..., 2:10, 20:28] = 1.0
    cam, _, _ = GradCAM(net, net[0])(x, class_idx=0)
    y, col = np.unravel_index(cam.argmax(), cam.shape)
    assert 2 <= y < 10 and 20 <= col < 28


def test_gradcam_works_inside_no_grad():
    m = calibrated_model()
    with torch.no_grad():
        cam, _, _ = GradCAM(m, m.features[-1])(torch.randn(1, 3, 96, 96))
    assert cam.shape == (96, 96)


def test_gradcam_remove_clears_hooks():
    m = build_model(pretrained=False)
    layer = m.features[-1]
    engine = GradCAM(m, layer)
    engine.remove()
    assert not layer._forward_hooks and not layer._backward_hooks


def test_class_weights_with_missing_class():
    w = class_weights(pd.DataFrame({"diagnosis": [0, 0, 2, 4]}))
    assert w.shape == (5,)
    assert w[1] == 0 and w[3] == 0
    assert w[2] > w[0] > 0


def test_train_transform_fills_rotation_with_gray():
    torch.manual_seed(0)
    proc = np.full((64, 64, 3), 128, np.uint8)
    tf = get_transforms(train=True, size=64)
    samples = [tf(proc) for _ in range(10)]
    assert samples[0].shape == (3, 64, 64)
    assert max(x.std() for x in samples) < 0.3  # cantos pretos da rotação elevariam muito o desvio


def test_load_rgb_missing_file_has_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="nao_existe"):
        load_rgb(tmp_path / "nao_existe.png")


def test_dataset_missing_image_names_the_file(tmp_path):
    ds = APTOSDataset(pd.DataFrame({"id_code": ["abc123"], "diagnosis": [0]}), tmp_path, False, 32)
    with pytest.raises(FileNotFoundError, match="abc123"):
        ds[0]


def test_qwk_keeps_grade_distance_with_missing_classes():
    # Calculado à mão com pesos (i-j)^2 sobre os 5 graus. Sem labels fixos, as classes
    # ausentes aproximam 1 e 4, e o resultado seria 0.
    assert qwk([0, 1, 4, 4], [1, 1, 4, 0]) == pytest.approx(7 / 24)


def test_compute_metrics_values():
    ys = np.array([0, 0, 2, 4])
    probs = np.array(
        [
            [0.9, 0.1, 0.0, 0.0, 0.0],
            [0.2, 0.7, 0.1, 0.0, 0.0],
            [0.1, 0.1, 0.8, 0.0, 0.0],
            [0.1, 0.0, 0.0, 0.1, 0.8],
        ]
    )
    m = compute_metrics(ys, probs)
    assert m["n"] == 4
    assert m["auc_referral"] == pytest.approx(1.0)
    assert m["recall_per_class"]["Sem RD"] == pytest.approx(0.5)
    assert m["confusion_matrix"][0] == [1, 1, 0, 0, 0]
    assert len(m["confusion_matrix"]) == 5


def test_compute_metrics_auc_undefined_with_one_side():
    probs = np.tile([0.9, 0.1, 0, 0, 0], (2, 1))
    assert compute_metrics(np.array([0, 1]), probs)["auc_referral"] is None
