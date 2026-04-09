"""
lipreader/src/train.py

Полный цикл обучения с:
  - MixUp аугментацией
  - Label Smoothing
  - Cosine Annealing с тёплым рестартом (CosineAnnealingWarmRestarts)
  - Ранней остановкой
  - TensorBoard логированием
  - Сохранением лучшей модели

Запуск:
    python -m src.train
    python -m src.train --epochs 100 --batch 8 --lr 1e-3
"""

import os
import sys
import argparse
import time
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, MODELS_DIR, CKPT_DIR, LOG_DIR,
    NUM_CLASSES, BATCH_SIZE, NUM_EPOCHS,
    LEARNING_RATE, LR_MIN, WEIGHT_DECAY,
    MIXUP_ALPHA, LABEL_SMOOTHING,
    EARLY_STOP_PATIENCE, SEED,
    IDX_TO_CLASS
)
from src.dataset import make_loaders
from src.model import LipReader, count_parameters


# ─────────────────────── Воспроизводимость ───────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ─────────────────────────── MixUp ───────────────────────────

def mixup_batch(x: torch.Tensor, y_oh: torch.Tensor,
                alpha: float) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Применяет MixUp к батчу.
    y_oh — one-hot метки (B, num_classes), float.
    Возвращает (mixed_x, mixed_y).
    """
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    b   = x.size(0)
    idx = torch.randperm(b, device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    mixed_y = lam * y_oh + (1 - lam) * y_oh[idx]
    return mixed_x, mixed_y


# ──────────────────────── Метрики ─────────────────────────────

def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


# ─────────────────────────── Эпоха ───────────────────────────

def run_epoch(model, loader, criterion_soft, criterion_hard,
              optimizer, device, is_train: bool,
              mixup_alpha: float = 0.0) -> tuple[float, float]:
    """
    Одна эпоха.
    criterion_soft — принимает soft-метки (для MixUp + Label Smoothing)
    criterion_hard — принимает long-метки (для метрики точности)
    Возвращает (avg_loss, accuracy).
    """
    model.train(is_train)
    total_loss = 0.0
    total_acc  = 0.0
    n_batches  = 0

    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for x, labels in tqdm(loader, leave=False,
                               desc="train" if is_train else "val"):
            x      = x.to(device)
            labels = labels.to(device)

            # One-hot для MixUp / Label Smoothing
            y_oh = torch.zeros(labels.size(0), NUM_CLASSES, device=device)
            y_oh.scatter_(1, labels.unsqueeze(1), 1.0)

            if is_train and mixup_alpha > 0:
                x, y_oh = mixup_batch(x, y_oh, mixup_alpha)

            logits = model(x)

            # Для лосса используем soft-метки
            loss = criterion_soft(logits, y_oh)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                # Gradient clipping — стабилизирует LSTM
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            # Точность считаем по hard-меткам
            acc = accuracy(logits, labels)

            total_loss += loss.item()
            total_acc  += acc
            n_batches  += 1

    return total_loss / n_batches, total_acc / n_batches


# ───────────────────────── Лосс с Label Smoothing ─────────────

class SoftCrossEntropy(nn.Module):
    """
    CrossEntropy для soft-меток (MixUp / Label Smoothing).
    """
    def __init__(self, smoothing: float = 0.0):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor,
                targets: torch.Tensor) -> torch.Tensor:
        """
        logits  : (B, C)
        targets : (B, C) soft или (B,) hard
        """
        log_prob = torch.log_softmax(logits, dim=-1)

        if targets.dim() == 1:
            # Hard labels → one-hot
            y = torch.zeros_like(log_prob)
            y.scatter_(1, targets.unsqueeze(1), 1.0)
        else:
            y = targets

        if self.smoothing > 0:
            n = logits.size(-1)
            y = y * (1 - self.smoothing) + self.smoothing / n

        return -(y * log_prob).sum(dim=-1).mean()


# ─────────────────────── Главная функция ─────────────────────

def train(data_dir: str = DATA_DIR,
          epochs: int = NUM_EPOCHS,
          batch_size: int = BATCH_SIZE,
          lr: float = LEARNING_RATE,
          lr_min: float = LR_MIN,
          weight_decay: float = WEIGHT_DECAY,
          mixup_alpha: float = MIXUP_ALPHA,
          label_smoothing: float = LABEL_SMOOTHING,
          patience: int = EARLY_STOP_PATIENCE):

    set_seed(SEED)
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR,   exist_ok=True)
    os.makedirs(LOG_DIR,    exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    # ── Данные ──────────────────────────────────────────────
    print("Загружаем датасет...")
    train_loader, val_loader, test_loader = make_loaders(
        data_dir, batch_size=batch_size, num_workers=2
    )

    # ── Модель ──────────────────────────────────────────────
    model = LipReader().to(device)
    print(f"Параметров: {count_parameters(model):,}")

    # ── Оптимизатор / Планировщик ───────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    # Cosine Annealing Warm Restarts (как в статье)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=lr_min
    )

    # ── Функции потерь ───────────────────────────────────────
    criterion = SoftCrossEntropy(smoothing=label_smoothing)

    # ── TensorBoard ──────────────────────────────────────────
    run_name = f"lipreader_{time.strftime('%Y%m%d_%H%M%S')}"
    writer   = SummaryWriter(os.path.join(LOG_DIR, run_name))

    # ── Ранняя остановка ─────────────────────────────────────
    best_val_acc  = 0.0
    best_epoch    = 0
    no_improve    = 0
    best_ckpt     = os.path.join(MODELS_DIR, "best_model.pt")

    print(f"\nНачинаем обучение ({epochs} эпох)...\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = run_epoch(
            model, train_loader, criterion, criterion,
            optimizer, device, is_train=True,
            mixup_alpha=mixup_alpha
        )
        val_loss, val_acc = run_epoch(
            model, val_loader, criterion, criterion,
            optimizer, device, is_train=False
        )

        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        # Логирование
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val",   val_loss,   epoch)
        writer.add_scalar("Acc/train",  train_acc,  epoch)
        writer.add_scalar("Acc/val",    val_acc,    epoch)
        writer.add_scalar("LR",         cur_lr,     epoch)

        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"Train loss={train_loss:.4f} acc={train_acc:.3f} | "
            f"Val loss={val_loss:.4f} acc={val_acc:.3f} | "
            f"LR={cur_lr:.2e} | {elapsed:.1f}s"
        )

        # Сохраняем лучшую модель
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            no_improve   = 0
            torch.save({
                "epoch":      epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "val_acc":    val_acc,
            }, best_ckpt)
            print(f"  ✓ Новый лучший результат: val_acc={val_acc:.4f}")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"\nРанняя остановка на эпохе {epoch} "
                      f"(лучший epoch={best_epoch}, val_acc={best_val_acc:.4f})")
                break

    writer.close()

    # ── Финальное тестирование ───────────────────────────────
    print(f"\nЗагружаем лучшую модель (epoch {best_epoch})...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state"])

    _, test_acc = run_epoch(
        model, test_loader, criterion, criterion,
        None, device, is_train=False
    )
    print(f"\n{'='*50}")
    print(f"  Test accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"{'='*50}")

    # Финальный экспорт
    final_path = os.path.join(MODELS_DIR, "lipreader_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"Модель сохранена: {final_path}")

    return model


# ──────────────────────────── CLI ────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение LipReader")
    parser.add_argument("--data",    default=DATA_DIR)
    parser.add_argument("--epochs",  type=int,   default=NUM_EPOCHS)
    parser.add_argument("--batch",   type=int,   default=BATCH_SIZE)
    parser.add_argument("--lr",      type=float, default=LEARNING_RATE)
    parser.add_argument("--mixup",   type=float, default=MIXUP_ALPHA)
    parser.add_argument("--smooth",  type=float, default=LABEL_SMOOTHING)
    parser.add_argument("--patience",type=int,   default=EARLY_STOP_PATIENCE)
    args = parser.parse_args()

    train(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        mixup_alpha=args.mixup,
        label_smoothing=args.smooth,
        patience=args.patience,
    )
