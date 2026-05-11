"""
Дообучение существующей модели на конкретных классах.
Запуск:
    python finetune.py                         # дообучить все классы
    python finetune.py --classes "Да,Пока"     # только нужные классы
"""

import os, sys, argparse, time, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    DATA_DIR, MODELS_DIR, CLASSES, CLASS_TO_IDX,
    BATCH_SIZE, SEED, NUM_CLASSES, LABEL_SMOOTHING
)
from src.dataset import make_splits, LipDataset, collect_samples
from src.train import SoftCrossEntropy, run_epoch, set_seed
from src.model import LipReader

# Параметры дообучения
FINETUNE_LR      = 2e-4   # низкий LR(скорость обучения) чтобы не забыть старое
FINETUNE_EPOCHS  = 40
FINETUNE_PATIENCE= 10
FOCUS_WEIGHT     = 4.0    # во сколько раз чаще показывать фокус-классы


def finetune(focus_classes: list[str]):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")
    print(f"Фокус-классы: {focus_classes}")

    # Загружаем существующую модель
    weights_path = os.path.join(MODELS_DIR, "lipreader_final.pt")
    if not os.path.exists(weights_path):
        # Пробуем best_model
        weights_path = os.path.join(MODELS_DIR, "best_model.pt")
    
    model = LipReader().to(device)
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    print(f"Модель загружена: {weights_path}")

    # Датасет с взвешенной выборкой
    train_ds, val_ds, _ = make_splits(DATA_DIR)

    # Веса для sampler: фокус-классы получают FOCUS_WEIGHT
    focus_idx = set(CLASS_TO_IDX[c] for c in focus_classes if c in CLASS_TO_IDX)
    sample_weights = []
    for path, label in train_ds.samples:
        w = FOCUS_WEIGHT if label in focus_idx else 1.0
        sample_weights.append(w)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE,
        sampler=sampler, num_workers=2, pin_memory=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=2, pin_memory=False
    )

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    focus_count = sum(1 for _, l in train_ds.samples if l in focus_idx)
    print(f"Из них фокус-классов: {focus_count} (показываем в {FOCUS_WEIGHT}x чаще)")

    # Оптимизатор 
    # Классификатор учится быстрее, frontend — медленно чтоб не сломать
    optimizer = torch.optim.AdamW([
        {"params": model.frontend.parameters(),    "lr": FINETUNE_LR * 0.1},
        {"params": model.spatial_pool.parameters(),"lr": FINETUNE_LR * 0.1},
        {"params": model.lstm.parameters(),        "lr": FINETUNE_LR * 0.5},
        {"params": model.classifier.parameters(),  "lr": FINETUNE_LR},
    ], weight_decay=1e-4)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=FINETUNE_EPOCHS, eta_min=1e-5
    )

    criterion = SoftCrossEntropy(smoothing=LABEL_SMOOTHING)

    best_val_acc = 0.0
    no_improve   = 0
    best_path    = os.path.join(MODELS_DIR, "lipreader_finetuned.pt")

    print(f"\nДообучение ({FINETUNE_EPOCHS} эпох)...\n")

    for epoch in range(1, FINETUNE_EPOCHS + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, criterion,
            optimizer, device, is_train=True, mixup_alpha=0.3
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, criterion,
            None, device, is_train=False
        )
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{FINETUNE_EPOCHS} | "
            f"Train acc={train_acc:.3f} | Val acc={val_acc:.3f} | "
            f"{elapsed:.0f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve   = 0
            torch.save(model.state_dict(), best_path)
            print(f"  ✓ Лучший результат: {val_acc:.4f}")
        else:
            no_improve += 1
            if no_improve >= FINETUNE_PATIENCE:
                print(f"\nРанняя остановка (лучший val_acc={best_val_acc:.4f})")
                break

    # Перезаписываем финальную модель
    final_path = os.path.join(MODELS_DIR, "lipreader_final.pt")
    import shutil
    shutil.copy(best_path, final_path)
    print(f"\n✓ Модель сохранена: {final_path}")
    print(f"  val_acc = {best_val_acc:.4f} ({best_val_acc*100:.2f}%)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--classes",
        default=",".join(CLASSES),
        help='Классы через запятую, например: "Да,Пока"'
    )
    args = parser.parse_args()
    focus = [c.strip() for c in args.classes.split(",") if c.strip()]
    finetune(focus)