"""Dataset APTOS 2019 + pré-processamento (Ben Graham) + augmentations."""
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
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
    img = cv2.resize(img, (size, size))
    blur = cv2.GaussianBlur(img, (0, 0), size / 30)
    return cv2.addWeighted(img, 4, blur, -4, 128)


def get_transforms(train: bool, size: int = 320):
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return T.Compose(
            [
                T.ToPILImage(),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(30),
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
        img = cv2.cvtColor(cv2.imread(str(self.img_dir / f"{row.id_code}.png")), cv2.COLOR_BGR2RGB)
        x = self.tf(preprocess(img, self.size))
        return x, int(row.diagnosis)


def make_splits(csv_path: Path, val_size: float = 0.2, seed: int = 42):
    df = pd.read_csv(csv_path)
    return train_test_split(df, test_size=val_size, stratify=df.diagnosis, random_state=seed)


def class_weights(df: pd.DataFrame) -> torch.Tensor:
    counts = df.diagnosis.value_counts().sort_index().values.astype(np.float32)
    w = counts.sum() / (len(counts) * counts)
    return torch.tensor(w)
