# 👄 LipReader

Нейросеть для чтения по губам — классификация 7 команд:
`bye · da · exit · hello · net · on · stop`

## Архитектура

```
Видео (.mp4)
    ↓
MediaPipe Face Mesh  →  область губ 112×112 (grayscale)
    ↓
3× ConvBlock3D (Conv3D → BN → ReLU → MaxPool3D)
    ↓
AdaptiveAvgPool (пространственное усреднение)
    ↓
BiLSTM × 2 слоя (512 нейронов)
    ↓
Dropout → FC(7) → Softmax
```

Вдохновлено: Аксёнов et al. (2022) «Метод визуального анализа лица водителя»

## Быстрый старт

### 1. Установка зависимостей
```bash
pip install -r requirements.txt
```

### 2. Структура данных
```
data/
├── bye/      ← ~50 видео .mp4
├── da/
├── exit/
├── hello/
├── net/
├── on/
└── stop/
```

### 3. Препроцессинг (извлечение губ)
```bash
python -m src.preprocess
```
Рядом с каждым `.mp4` появится `.npy` файл (32 кадра губ, нормализованных).

Принудительная перезапись:
```bash
python -m src.preprocess --force
```

### 4. Обучение
```bash
python -m src.train
```

Параметры:
```bash
python -m src.train --epochs 80 --batch 8 --lr 1e-3 --mixup 0.4
```

Мониторинг в TensorBoard:
```bash
tensorboard --logdir runs/
```

### 5. Оценка модели
```bash
python -m src.evaluate
```
Выводит точность по классам + confusion matrix (`models/confusion_matrix.png`).

### 6. Инференс на видео
```bash
python predict.py path/to/video.mp4
python predict.py path/to/video.mp4 --top3
```

## Конфигурация

Все гиперпараметры — в `config.py`:

| Параметр           | Значение | Описание                          |
|--------------------|----------|-----------------------------------|
| `SEQUENCE_LENGTH`  | 32       | Кадров на видео                   |
| `FRAME_W/H`        | 112      | Размер кадра губ                  |
| `BATCH_SIZE`       | 8        | Размер батча                      |
| `NUM_EPOCHS`       | 80       | Максимум эпох                     |
| `LEARNING_RATE`    | 1e-3     | Начальный LR                      |
| `MIXUP_ALPHA`      | 0.4      | Сила MixUp аугментации            |
| `LABEL_SMOOTHING`  | 0.1      | Сглаживание меток                 |
| `LSTM_HIDDEN`      | 256      | Нейронов в BiLSTM                 |
| `EARLY_STOP_PATIENCE` | 12    | Эпох без улучшения до остановки   |

## Структура проекта

```
lipreader/
├── config.py          # все гиперпараметры
├── predict.py         # инференс на одном видео
├── requirements.txt
├── src/
│   ├── preprocess.py  # MediaPipe: видео → .npy
│   ├── dataset.py     # PyTorch Dataset / DataLoader
│   ├── model.py       # Conv3D + BiLSTM
│   ├── train.py       # цикл обучения
│   └── evaluate.py    # метрики + confusion matrix
├── models/            # сохранённые веса
│   ├── best_model.pt
│   └── lipreader_final.pt
├── checkpoints/       # чекпоинты во время обучения
└── runs/              # TensorBoard логи
```

## Советы при маленьком датасете (~50 видео на класс)

- `USE_HFLIP = True` — горизонтальный флип удваивает данные
- `MIXUP_ALPHA = 0.4` — смешивает примеры, уменьшает переобучение
- `LABEL_SMOOTHING = 0.1` — модель не «зазубривает» метки
- `EARLY_STOP_PATIENCE = 12` — остановка при переобучении
- При GPU ускорение ~10× по сравнению с CPU
