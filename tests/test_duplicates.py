import json
import shutil

import cv2
import numpy as np
import pandas as pd

from scripts.check_duplicates import cross_split_distances, main, phash, vessel_signature
from scripts.make_synthetic_data import generate
from src.dr import train
from src.dr.data import make_splits


def fundus(seed: int, size: int = 512) -> np.ndarray:
    """Disco com textura suave própria, para que o hash tenha estrutura a comparar."""
    rng = np.random.default_rng(seed)
    noise = cv2.resize(rng.random((8, 8, 3)).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    img = np.uint8(np.clip(noise * 255, 8, 255))
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (size // 2, size // 2), size // 2 - 8, 255, -1)
    img[mask == 0] = 0
    return img


def bits(a, b) -> int:
    return int((phash(a) != phash(b)).sum())


def test_phash_matches_resized_and_brighter_copies():
    img = fundus(1)
    assert bits(img, cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)) <= 4
    assert bits(img, cv2.convertScaleAbs(img, alpha=1.0, beta=12)) <= 4


def test_phash_ignores_black_border_padding():
    img = fundus(2)
    assert bits(img, cv2.copyMakeBorder(img, 0, 0, 100, 100, cv2.BORDER_CONSTANT, value=0)) <= 4


def test_phash_separates_different_images():
    assert bits(fundus(3), fundus(4)) > 8


def textured(seed: int, size: int = 512) -> np.ndarray:
    """Como fundus(), mas com detalhes finos, que é o que a assinatura compara."""
    rng = np.random.default_rng(seed)
    noise = cv2.resize(rng.random((48, 48, 3)).astype(np.float32), (size, size), interpolation=cv2.INTER_CUBIC)
    img = np.uint8(np.clip(noise * 255, 8, 255))
    mask = np.zeros((size, size), np.uint8)
    cv2.circle(mask, (size // 2, size // 2), size // 2 - 8, 255, -1)
    img[mask == 0] = 0
    return img


def test_signature_high_for_altered_copies_and_low_for_other_images():
    img = textured(5)
    sig = vessel_signature(img)
    assert sig @ vessel_signature(img) > 0.999
    assert sig @ vessel_signature(cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)) > 0.9
    # beta fica abaixo da tolerância do recorte (7). Acima dela a borda não é cortada e a cópia
    # passa despercebida, então o método dá um piso do número de duplicatas.
    assert sig @ vessel_signature(cv2.convertScaleAbs(img, alpha=0.85, beta=5)) > 0.9
    assert abs(sig @ vessel_signature(textured(6))) < 0.3


def test_signature_ignores_shared_disk_outline():
    """Um disco liso e um disco com textura têm o mesmo contorno, mas não são a mesma imagem."""
    plain = np.zeros((512, 512, 3), np.uint8)
    cv2.circle(plain, (256, 256), 248, (180, 90, 40), -1)
    assert abs(vessel_signature(plain) @ vessel_signature(textured(7))) < 0.3


def test_distance_matrix_shape_and_values():
    val = np.array([[True, True, False], [False, False, False]])
    train_hashes = np.array([[True, True, False], [True, True, True], [False, True, False]])
    expected = [[0, 1, 1], [2, 3, 1]]
    assert cross_split_distances(val, train_hashes).tolist() == expected


def test_report_finds_planted_duplicate(tmp_path):
    data = tmp_path / "aptos"
    generate(data, per_class=6, size=96)
    weights = tmp_path / "best.pt"
    train.main(
        ["--data", str(data), "--epochs", "1", "--bs", "8", "--size", "64",
         "--workers", "0", "--no-pretrained", "--out", str(weights)]
    )
    # Copia uma imagem da validação sobre uma do treino, com rótulos diferentes.
    tr_df, va_df = make_splits(data / "train.csv")
    src = va_df[va_df.diagnosis == 4].id_code.iloc[0]
    dst = tr_df[tr_df.diagnosis == 0].id_code.iloc[0]
    shutil.copy(data / "train_images" / f"{src}.png", data / "train_images" / f"{dst}.png")

    out = tmp_path / "report.json"
    main(["--data", str(data), "--weights", str(weights), "--size", "64", "--workers", "0", "--out", str(out)])

    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["n_val"] == 6
    assert report["by_threshold"][0]["max_bit_distance"] == 0

    pairs = pd.read_csv(data / "duplicate_pairs.csv")
    planted = pairs[(pairs.val_id == src) & (pairs.train_id == dst)]
    assert len(planted) == 1
    assert planted.bit_distance.iloc[0] == 0
    assert planted.correlation.iloc[0] > 0.999
    assert planted.val_label.iloc[0] == 4 and planted.train_label.iloc[0] == 0
    assert report["by_correlation"][-1]["val_images_with_match"] >= 1
