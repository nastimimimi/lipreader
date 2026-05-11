# 👄 LipReader

A neural network for lip reading — classifying 7 spoken commands from video:

`bye · da · exit · hello · net · on · stop`

Built with Conv3D + BiLSTM architecture and MediaPipe for lip region extraction.
Inspired by Aksenov et al. (2022) *"Visual Analysis Method for Driver's Face"*.

---

## Architecture

```
Video (.mp4)
    ↓
MediaPipe Face Mesh  →  lip region 112×112 (grayscale)
    ↓
3× ConvBlock3D (Conv3D → BN → ReLU → MaxPool3D)
    ↓
AdaptiveAvgPool (spatial averaging)
    ↓
BiLSTM × 2 layers (512 neurons)
    ↓
Dropout → FC(7) → Softmax
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Data structure

```
data/
├── bye/      ← ~50 .mp4 videos per class
├── da/
├── exit/
├── hello/
├── net/
├── on/
└── stop/
```

### 3. Preprocessing

Extract lip regions from all videos:

```bash
python -m src.preprocess
```

A `.npy` file (32 normalised lip frames) will be created next to each `.mp4`.
To force reprocess existing files:

```bash
python -m src.preprocess --force
```

### 4. Training

```bash
python -m src.train
```

Custom parameters:

```bash
python -m src.train --epochs 80 --batch 8 --lr 1e-3 --mixup 0.4
```

Monitor training in TensorBoard:

```bash
tensorboard --logdir runs/
```

### 5. Evaluate

```bash
python -m src.evaluate
```

Outputs per-class accuracy and a confusion matrix saved to `models/confusion_matrix.png`.

### 6. Inference

Run prediction on a single video:

```bash
python predict.py path/to/video.mp4
```

Show top-3 predictions:

```bash
python predict.py path/to/video.mp4 --top3
```

---

## Configuration

All hyperparameters are defined in `config.py`:

| Parameter              | Value | Description                            |
|------------------------|-------|----------------------------------------|
| `SEQUENCE_LENGTH`      | 32    | Frames per video                       |
| `FRAME_W/H`            | 112   | Lip frame size (px)                    |
| `BATCH_SIZE`           | 8     | Batch size                             |
| `NUM_EPOCHS`           | 80    | Maximum training epochs                |
| `LEARNING_RATE`        | 1e-3  | Initial learning rate                  |
| `MIXUP_ALPHA`          | 0.4   | MixUp augmentation strength            |
| `LABEL_SMOOTHING`      | 0.1   | Label smoothing factor                 |
| `LSTM_HIDDEN`          | 256   | Neurons per BiLSTM layer               |
| `EARLY_STOP_PATIENCE`  | 12    | Epochs without improvement before stop |

---

## Project Structure

```
lipreader/
├── config.py           # all hyperparameters
├── predict.py          # single video inference
├── requirements.txt
├── src/
│   ├── preprocess.py   # MediaPipe: video → .npy
│   ├── dataset.py      # PyTorch Dataset / DataLoader
│   ├── model.py        # Conv3D + BiLSTM definition
│   ├── train.py        # training loop
│   └── evaluate.py     # metrics + confusion matrix
├── models/
│   ├── best_model.pt
│   └── lipreader_final.pt
├── checkpoints/        # training checkpoints
└── runs/               # TensorBoard logs
```

---

## Requirements

- Python 3.9+
- PyTorch 2.0+
- MediaPipe
- OpenCV
- NumPy

---

*Made by Nastya Baranova · Odessa, Ukraine*
