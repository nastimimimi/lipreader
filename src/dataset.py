"""
lipreader/src/dataset.py

PyTorch Dataset для чтения по губам.
Читает .npy файлы (созданные preprocess.py) и отдаёт тензоры.
"""

import os
import sys
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, CLASSES, CLASS_TO_IDX,
    SEQUENCE_LENGTH, FRAME_H, FRAME_W,
    TRAIN_RATIO, VAL_RATIO,
    USE_HFLIP, SEED
)


class LipDataset(Dataset):
    """
    Датасет для классификации по губам.

    Args:
        samples  : список (npy_path, class_idx)
        augment  : применять ли аугментацию (горизонтальный флип)
    """

    def __init__(self, samples: list[tuple[str, int]], augment: bool = False):
        self.samples = samples
        self.augment = augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]

        # shape: (T, H, W)  float32, уже нормализован
        frames = np.load(path)

        # Убеждаемся что длина ровно SEQUENCE_LENGTH
        frames = _fix_length(frames, SEQUENCE_LENGTH)

        # Аугментация: горизонтальный флип
        if self.augment and USE_HFLIP and random.random() < 0.5:
            frames = frames[:, :, ::-1].copy()

        # (T, H, W) → (1, T, H, W)  — канал для Conv3D
        tensor = torch.from_numpy(frames).unsqueeze(0)  # float32

        return tensor, label


def _fix_length(frames: np.ndarray, target: int) -> np.ndarray:
    """Обрезает или дополняет последовательность до target кадров."""
    t = frames.shape[0]
    if t == target:
        return frames
    if t > target:
        indices = np.linspace(0, t - 1, target, dtype=int)
        return frames[indices]
    # Дополняем последним кадром
    pad = np.stack([frames[-1]] * (target - t), axis=0)
    return np.concatenate([frames, pad], axis=0)


# ────────────────────── Сбор файлов ───────────────────────────

def collect_samples(data_dir: str) -> list[tuple[str, int]]:
    """
    Обходит data_dir, возвращает список (npy_path, class_idx).
    """
    samples = []
    for cls in CLASSES:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        label = CLASS_TO_IDX[cls]
        for fname in os.listdir(cls_dir):
            if fname.endswith(".npy"):
                samples.append((os.path.join(cls_dir, fname), label))
    return samples


def make_splits(data_dir: str = DATA_DIR
                ) -> tuple[LipDataset, LipDataset, LipDataset]:
    """
    Делит датасет на train / val / test, возвращает три Dataset-объекта.
    Стратифицированный сплит (по классам).
    """
    samples = collect_samples(data_dir)
    if len(samples) == 0:
        raise RuntimeError(
            f"Не найдено ни одного .npy файла в {data_dir}.\n"
            "Сначала запусти: python -m src.preprocess"
        )

    labels  = [s[1] for s in samples]

    # train vs. (val + test)
    train_s, tmp_s, _, tmp_l = train_test_split(
        samples, labels,
        test_size=1 - TRAIN_RATIO,
        stratify=labels,
        random_state=SEED,
    )
    # val vs. test
    val_ratio_adj = VAL_RATIO / (1 - TRAIN_RATIO)
    val_s, test_s = train_test_split(
        tmp_s,
        test_size=1 - val_ratio_adj,
        stratify=tmp_l,
        random_state=SEED,
    )

    print(f"  Train: {len(train_s)} | Val: {len(val_s)} | Test: {len(test_s)}")

    return (
        LipDataset(train_s, augment=True),
        LipDataset(val_s,   augment=False),
        LipDataset(test_s,  augment=False),
    )


def make_loaders(data_dir: str = DATA_DIR,
                 batch_size: int = 8,
                 num_workers: int = 2
                 ) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Удобная обёртка: возвращает три DataLoader.
    """
    train_ds, val_ds, test_ds = make_splits(data_dir)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        shuffle=True, num_workers=num_workers,
        pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size,
        shuffle=False, num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader


# ──────────────────── Быстрая проверка ────────────────────────
if __name__ == "__main__":
    train_l, val_l, test_l = make_loaders()
    x, y = next(iter(train_l))
    print(f"Batch shape : {x.shape}")   # (B, 1, T, H, W)
    print(f"Labels      : {y}")
    print(f"dtype / min / max: {x.dtype} / {x.min():.2f} / {x.max():.2f}")
