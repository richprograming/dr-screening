"""Dataset APTOS 2019 + pré-processamento (Ben Graham) + augmentations."""
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

CLASSES = ["Sem RD", "Leve", "Moderada", "Grave", "Proliferativa"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def crop_black_borders(img: np.ndarray, tol: int = 7) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if mask.sum() == 0:
        return img
    ys, xs = np.where(mask)
    return img[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]


def preprocess(img: np.ndarray, size: int = 320) -> np.ndarray:
    """Recorta bordas pretas, redimensiona e realça lesões (Ben Graham)."""
    img = crop_black_borders(img)
    # INTER_AREA evita aliasing ao reduzir: lesões pequenas somem com INTER_LINEAR.
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    blur = cv2.GaussianBlur(img, (0, 0), size / 30)
    return cv2.addWeighted(img, 4, blur, -4, 128)


def load_rgb(path: Path) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:  # cv2.imread não levanta exceção para arquivo ausente ou corrompido
        raise FileNotFoundError(f"Não foi possível ler a imagem: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def get_transforms(train: bool, size: int = 320):
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return T.Compose(
            [
                T.ToPILImage(),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                # fill=128 é o fundo cinza do Ben Graham; o padrão (preto) criaria
                # cantos que só existem no treino.
                T.RandomRotation(30, fill=128),
                T.ColorJitter(0.1, 0.1),
                T.ToTensor(),
                norm,
            ]
        )
    return T.Compose([T.ToPILImage(), T.ToTensor(), norm])


class APTOSDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: Path, train: bool, size: int = 320):
        self.df, self.img_dir, self.size = df.reset_index(drop=True), Path(img_dir), size
        self.tf = get_transforms(train, size)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        img = load_rgb(self.img_dir / f"{row.id_code}.png")
        x = self.tf(preprocess(img, self.size))
        return x, int(row.diagnosis)


def make_splits(csv_path: Path, val_size: float = 0.2, seed: int = 42, limit: int | None = None):
    """Divisão estratificada. `limit` usa um subconjunto estratificado (só para testes rápidos)."""
    df = pd.read_csv(csv_path)
    if limit is not None and limit < len(df):
        df, _ = train_test_split(df, train_size=limit, stratify=df.diagnosis, random_state=seed)
    return train_test_split(df, test_size=val_size, stratify=df.diagnosis, random_state=seed)


def class_weights(df: pd.DataFrame) -> torch.Tensor:
    # reindex garante um peso por classe mesmo se alguma não aparecer no subconjunto.
    counts = df.diagnosis.value_counts().reindex(range(len(CLASSES)), fill_value=0).values.astype(np.float32)
    w = np.divide(counts.sum(), len(counts) * counts, out=np.zeros_like(counts), where=counts > 0)
    return torch.tensor(w)


def default_num_workers() -> int:
    """Windows e macOS usam `spawn`: cada worker reimporta torch e cv2, o que costuma
    ser lento e frágil. Nesses sistemas o padrão é 0; use --workers para sobrescrever."""
    if sys.platform in ("win32", "darwin"):
        return 0
    return min(4, os.cpu_count() or 1)


def _init_worker(_: int) -> None:
    cv2.setNumThreads(0)  # evita threads do OpenCV competindo com os workers do DataLoader


def make_loader(ds: Dataset, batch_size: int, train: bool = False, workers: int | None = None) -> DataLoader:
    workers = default_num_workers() if workers is None else workers
    return DataLoader(
        ds,
        batch_size,
        shuffle=train,
        # Um lote de tamanho 1 no treino dá estatísticas ruins ao BatchNorm.
        drop_last=train and len(ds) % batch_size == 1,
        num_workers=workers,
        persistent_workers=workers > 0,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=_init_worker if workers > 0 else None,
    )
