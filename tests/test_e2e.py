"""Ponta a ponta com imagens sintéticas e sem baixar nada: treino, avaliação, exportação,
API e demos. Confirma que as peças se encaixam, não a qualidade do modelo."""
import base64
import json

import cv2
import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from scripts import gradcam_demo
from scripts.make_synthetic_data import generate, make_image
from src.dr import evaluate, export_onnx, train
from src.dr.data import CLASSES, make_splits


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    data = generate(root / "aptos", per_class=6, size=96)
    weights = root / "nested" / "dir" / "best.pt"  # o diretório ainda não existe
    train.main(
        ["--data", str(data), "--epochs", "1", "--bs", "8", "--size", "64",
         "--workers", "0", "--no-pretrained", "--out", str(weights)]
    )
    assert weights.exists()
    return root, data, weights


def png_bytes(grade=2):
    ok, buf = cv2.imencode(".png", make_image(grade, np.random.default_rng(0), 128))
    assert ok
    return buf.tobytes()


def test_limit_is_stratified_and_reproducible(tmp_path):
    data = generate(tmp_path / "d", per_class=10, size=32)
    a_tr, a_va = make_splits(data / "train.csv", limit=25)
    b_tr, b_va = make_splits(data / "train.csv", limit=25)
    assert len(a_tr) + len(a_va) == 25
    assert list(a_va.id_code) == list(b_va.id_code)
    assert a_va.diagnosis.nunique() == 5


def test_evaluate_writes_metrics(trained, tmp_path):
    _, data, weights = trained
    evaluate.main(
        ["--data", str(data), "--weights", str(weights), "--size", "64",
         "--workers", "0", "--out-dir", str(tmp_path)]
    )
    m = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert m["n"] == 6
    assert np.array(m["confusion_matrix"]).sum() == 6
    assert (tmp_path / "confusion_matrix.png").stat().st_size > 0


def test_onnx_matches_pytorch(trained, tmp_path):
    ort = pytest.importorskip("onnxruntime")
    from src.dr.model import load_model

    _, _, weights = trained
    out = tmp_path / "sub" / "dr.onnx"
    export_onnx.main(["--weights", str(weights), "--out", str(out), "--size", "64"])
    assert not out.with_suffix(".onnx.data").exists()

    x = torch.randn(2, 3, 64, 64)  # lote diferente do usado na exportação
    session = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    got = session.run(None, {"image": x.numpy()})[0]
    with torch.no_grad():
        expected = load_model(weights)(x).numpy()
    assert np.allclose(got, expected, atol=1e-3)


def test_gradcam_demo_saves_panels(trained, tmp_path):
    _, data, weights = trained
    gradcam_demo.main(["--data", str(data), "--weights", str(weights), "--out", str(tmp_path), "--size", "64"])
    assert len(list(tmp_path.glob("gradcam_grau*.png"))) == 5


def test_api_with_weights(trained, monkeypatch):
    from api.main import app

    monkeypatch.setenv("DR_WEIGHTS", str(trained[2]))
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "model_loaded": True}

        r = client.post("/predict", files={"file": ("a.png", png_bytes(), "image/png")})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["label"] == CLASSES[body["grade"]]
        assert body["refer"] == (body["grade"] >= 2)
        assert sum(body["probabilities"].values()) == pytest.approx(1, abs=1e-2)
        assert "Não é dispositivo médico" in body["disclaimer"]
        overlay = cv2.imdecode(np.frombuffer(base64.b64decode(body["gradcam_png_base64"]), np.uint8), cv2.IMREAD_COLOR)
        assert overlay.shape == (320, 320, 3)

        for raw in (b"", b"isto nao e uma imagem"):
            assert client.post("/predict", files={"file": ("a.png", raw, "image/png")}).status_code == 400


def test_api_rejects_oversized_upload(trained, monkeypatch):
    import api.main

    monkeypatch.setenv("DR_WEIGHTS", str(trained[2]))
    monkeypatch.setattr(api.main, "MAX_UPLOAD_BYTES", 10)
    with TestClient(api.main.app) as client:
        assert client.post("/predict", files={"file": ("a.png", png_bytes(), "image/png")}).status_code == 413


def test_api_without_weights_starts_and_answers_503(tmp_path, monkeypatch):
    from api.main import app

    monkeypatch.setenv("DR_WEIGHTS", str(tmp_path / "nao_existe.pt"))
    with TestClient(app) as client:
        assert client.get("/health").json() == {"status": "ok", "model_loaded": False}
        assert client.post("/predict", files={"file": ("a.png", png_bytes(), "image/png")}).status_code == 503


def test_gradio_app(trained, monkeypatch):
    gr = pytest.importorskip("gradio")
    import app as demo_app

    monkeypatch.setattr(demo_app, "_predictor", None)
    monkeypatch.setenv("DR_WEIGHTS", str(trained[2]))
    monkeypatch.delenv("DR_WEIGHTS_REPO", raising=False)
    probs, processed, heat, summary = demo_app.classify(cv2.cvtColor(make_image(3, np.random.default_rng(1)), cv2.COLOR_BGR2RGB))
    assert set(probs) == set(CLASSES)
    assert processed.shape == heat.shape == (320, 320, 3)
    assert "Grau mais provável" in summary

    with pytest.raises(gr.Error):
        demo_app.classify(None)

    monkeypatch.setattr(demo_app, "_predictor", None)
    monkeypatch.setenv("DR_WEIGHTS", "nao_existe.pt")
    with pytest.raises(gr.Error, match="Pesos não encontrados"):
        demo_app.classify(np.zeros((32, 32, 3), np.uint8))
