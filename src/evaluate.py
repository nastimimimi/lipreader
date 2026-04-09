"""
lipreader/src/evaluate.py

Детальная оценка модели:
  - Общая точность на тестовом множестве
  - Confusion matrix
  - Точность по каждому классу
  - Топ-2 ошибок (какие классы чаще всего путаются)

Запуск:
    python -m src.evaluate
    python -m src.evaluate --weights models/lipreader_final.pt
"""

import os
import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DATA_DIR, MODELS_DIR, CLASSES, NUM_CLASSES, BATCH_SIZE
from src.dataset import make_loaders
from src.model import LipReader


def evaluate(weights_path: str, data_dir: str = DATA_DIR):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем модель
    model = LipReader().to(device)
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    print(f"Модель загружена: {weights_path}")

    # Загружаем тестовый датасет
    _, _, test_loader = make_loaders(data_dir, batch_size=BATCH_SIZE)

    all_preds  = []
    all_labels = []

    with torch.no_grad():
        for x, labels in test_loader:
            x      = x.to(device)
            logits = model(x)
            preds  = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    # ── Общая точность ──────────────────────────────────────
    total_acc = (all_preds == all_labels).mean()
    print(f"\nОбщая точность: {total_acc*100:.2f}%  "
          f"({(all_preds == all_labels).sum()}/{len(all_labels)})")

    # ── Classification report ────────────────────────────────
    print("\n" + classification_report(
        all_labels, all_preds, target_names=CLASSES
    ))

    # ── Confusion matrix ─────────────────────────────────────
    cm = confusion_matrix(all_labels, all_preds)
    _plot_confusion_matrix(cm, CLASSES)

    # ── Топ ошибок ───────────────────────────────────────────
    _print_top_errors(cm)

    return total_acc


def _plot_confusion_matrix(cm: np.ndarray, class_names: list):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=range(len(class_names)),
        yticks=range(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="Истинный класс",
        xlabel="Предсказанный класс",
        title="Матрица ошибок (Confusion Matrix)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=11)

    # Подписи в ячейках
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)

    fig.tight_layout()
    out_path = os.path.join(MODELS_DIR, "confusion_matrix.png")
    fig.savefig(out_path, dpi=150)
    print(f"Confusion matrix сохранена: {out_path}")
    plt.close(fig)


def _print_top_errors(cm: np.ndarray):
    print("\nЧаще всего путаются пары:")
    errors = []
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            if i != j and cm[i, j] > 0:
                errors.append((cm[i, j], CLASSES[i], CLASSES[j]))
    errors.sort(reverse=True)
    for count, true_cls, pred_cls in errors[:5]:
        print(f"  '{true_cls}' → '{pred_cls}' : {count} раз")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Оценка модели")
    parser.add_argument(
        "--weights",
        default=os.path.join(MODELS_DIR, "lipreader_final.pt"),
        help="Путь к файлу весов .pt"
    )
    parser.add_argument("--data", default=DATA_DIR)
    args = parser.parse_args()

    evaluate(args.weights, args.data)
