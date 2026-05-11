"""
lipreader/config.py
Все гиперпараметры и настройки проекта в одном месте.
"""

import os

# ───────────────────────────── ПУТИ ─────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "data")
MODELS_DIR  = os.path.join(BASE_DIR, "models")
CKPT_DIR    = os.path.join(BASE_DIR, "checkpoints")
LOG_DIR     = os.path.join(BASE_DIR, "runs")

# ──────────────────────────── КЛАССЫ ────────────────────────────
CLASSES     = ["Пока", "Да", "Уйди", "Привет", "Нет", "Включи", "Стоп"]
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
IDX_TO_CLASS = {i: c for c, i in CLASS_TO_IDX.items()}

# ───────────────────────── ПРЕПРОЦЕССИНГ ────────────────────────
FRAME_W = 112
FRAME_H = 112
SEQUENCE_LENGTH = 32
TARGET_FPS = 25

# ─────────────────────────── МОДЕЛЬ ─────────────────────────────
LSTM_HIDDEN  = 128
LSTM_LAYERS  = 1
LSTM_DROPOUT = 0.5
FC_DROPOUT   = 0.5

# ─────────────────────────── ОБУЧЕНИЕ ───────────────────────────
BATCH_SIZE    = 8
NUM_EPOCHS    = 100
LEARNING_RATE = 1e-3
LR_MIN        = 1e-4
WEIGHT_DECAY  = 1e-3

TRAIN_RATIO = 0.7
VAL_RATIO   = 0.15

USE_HFLIP        = True
MIXUP_ALPHA      = 0.6
LABEL_SMOOTHING  = 0.2

EARLY_STOP_PATIENCE = 20

SEED = 42

# ─────────────────────── ИНФЕРЕНС ───────────────────────────────
CONFIDENCE_THRESHOLD = 0.5