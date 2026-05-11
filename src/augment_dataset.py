"""
lipreader/src/augment_dataset.py

Расширяет датасет из существующих .npy файлов:
  - горизонтальный флип
  - небольшой временной сдвиг
  - яркостный шум
  - временное растяжение/сжатие

Запуск:
    python -m src.augment_dataset
"""

import os, sys
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, CLASSES, SEQUENCE_LENGTH

def augment_hflip(frames):
    """Горизонтальный флип."""
    return frames[:, :, ::-1].copy()

def augment_brightness(frames):
    """Случайное изменение яркости."""
    factor = np.random.uniform(0.8, 1.2)
    return np.clip(frames * factor, frames.min(), frames.max())

def augment_time_shift(frames, max_shift=4):
    """Сдвиг по времени с заполнением крайним кадром."""
    shift = np.random.randint(1, max_shift + 1)
    direction = np.random.choice([-1, 1])
    result = np.roll(frames, shift * direction, axis=0)
    if direction > 0:
        result[:shift] = frames[0]
    else:
        result[-shift:] = frames[-1]
    return result

def augment_time_stretch(frames, factor_range=(0.85, 1.15)):
    """Лёгкое временное растяжение/сжатие через ресэмплинг."""
    factor = np.random.uniform(*factor_range)
    t = frames.shape[0]
    new_t = int(t * factor)
    new_t = max(new_t, SEQUENCE_LENGTH)
    indices = np.linspace(0, t - 1, new_t)
    stretched = np.array([frames[int(i)] for i in indices])
    # Обрезаем или дополняем до SEQUENCE_LENGTH
    if len(stretched) >= SEQUENCE_LENGTH:
        start = np.random.randint(0, len(stretched) - SEQUENCE_LENGTH + 1)
        return stretched[start:start + SEQUENCE_LENGTH]
    pad = np.stack([stretched[-1]] * (SEQUENCE_LENGTH - len(stretched)))
    return np.concatenate([stretched, pad], axis=0)

def augment_noise(frames):
    """Гауссов шум."""
    noise = np.random.normal(0, 0.05, frames.shape).astype(np.float32)
    return frames + noise


AUGMENTATIONS = [
    ("flip",    augment_hflip),
    ("bright",  augment_brightness),
    ("tshift",  augment_time_shift),
    ("stretch", augment_time_stretch),
    ("noise",   augment_noise),
]


def augment_dataset(data_dir=DATA_DIR, force=False):
    total_created = 0

    for cls in CLASSES:
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            continue

        # Только оригинальные файлы (без суффиксов аугментации)
        originals = [f for f in os.listdir(cls_dir)
                     if f.endswith(".npy") and "_aug_" not in f]

        print(f"\n[{cls}] {len(originals)} оригиналов")

        for fname in tqdm(originals, desc=cls):
            base = os.path.splitext(fname)[0]
            path = os.path.join(cls_dir, fname)
            frames = np.load(path)

            for aug_name, aug_fn in AUGMENTATIONS:
                out_name = f"{base}_aug_{aug_name}.npy"
                out_path = os.path.join(cls_dir, out_name)

                if os.path.exists(out_path) and not force:
                    continue

                augmented = aug_fn(frames).astype(np.float32)
                np.save(out_path, augmented)
                total_created += 1

    print(f"\nСоздано новых файлов: {total_created}")
    print(f"Итого .npy в датасете: {sum(1 for c in CLASSES for f in os.listdir(os.path.join(data_dir, c)) if f.endswith('.npy'))}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    augment_dataset(force=args.force)
