"""
lipreader/src/preprocess.py

Извлечение области губ из видео с помощью MediaPipe FaceLandmarker (новый API 0.10+).
Сохраняет последовательность кадров (grayscale) в .npy файл рядом с видео.

Использование:
    python -m src.preprocess              # обработать весь data/
    python -m src.preprocess --force      # перезаписать уже обработанные
"""

import os
import sys
import argparse
import urllib.request
import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    DATA_DIR, FRAME_W, FRAME_H,
    SEQUENCE_LENGTH, CLASSES
)

# ──────────────────── MediaPipe новый API ──────────────────────
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe.tasks.python.vision import FaceLandmarker, FaceLandmarkerOptions, RunningMode

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models", "face_landmarker.task"
)
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"


def _ensure_model():
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_PATH):
        print(f"Скачиваем модель MediaPipe -> {MODEL_PATH}")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("  OK")


LIP_INDICES = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
    78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
    308, 415, 310, 311, 312, 13, 82, 81, 80, 191,
]
LIP_PADDING = 0.4


def _crop_lip(frame_bgr, landmarks):
    h, w = frame_bgr.shape[:2]
    xs = [landmarks[i].x * w for i in LIP_INDICES]
    ys = [landmarks[i].y * h for i in LIP_INDICES]
    x_min, x_max = int(min(xs)), int(max(xs))
    y_min, y_max = int(min(ys)), int(max(ys))
    bw = max(x_max - x_min, 1)
    bh = max(y_max - y_min, 1)
    pad_x = int(bw * LIP_PADDING)
    pad_y = int(bh * LIP_PADDING)
    x1 = max(0, x_min - pad_x)
    y1 = max(0, y_min - pad_y)
    x2 = min(w, x_max + pad_x)
    y2 = min(h, y_max + pad_y)
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    return cv2.resize(gray, (FRAME_W, FRAME_H))


def video_to_frames(video_path, landmarker):
    import mediapipe as mp
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  [WARN] Не удалось открыть: {video_path}")
        return None

    frames_raw = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = landmarker.detect(mp_image)
        if result.face_landmarks:
            lip = _crop_lip(frame_bgr, result.face_landmarks[0])
            if lip is not None:
                frames_raw.append(lip)
    cap.release()

    if len(frames_raw) == 0:
        print(f"  [WARN] Губы не найдены: {video_path}")
        return None

    frames = _sample_frames(frames_raw, SEQUENCE_LENGTH)
    return np.stack(frames, axis=0).astype(np.float32)


def _sample_frames(frames, target_len):
    n = len(frames)
    if n == target_len:
        return frames
    if n > target_len:
        indices = np.linspace(0, n - 1, target_len, dtype=int)
        return [frames[i] for i in indices]
    return frames + [frames[-1]] * (target_len - n)


def normalize(frames):
    mean = frames.mean()
    std  = frames.std() + 1e-6
    return (frames - mean) / std


def _make_landmarker():
    options = FaceLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4,
    )
    return FaceLandmarker.create_from_options(options)


def process_dataset(data_dir, force=False):
    _ensure_model()
    total_ok = 0
    total_fail = 0

    with _make_landmarker() as landmarker:
        for cls in CLASSES:
            cls_dir = os.path.join(data_dir, cls)
            if not os.path.isdir(cls_dir):
                print(f"[SKIP] Папка не найдена: {cls_dir}")
                continue

            videos = [f for f in os.listdir(cls_dir)
                      if f.lower().endswith((".mp4", ".avi", ".mov", ".mpg"))]

            print(f"\n[{cls}] {len(videos)} видео")
            for vid in tqdm(videos, desc=cls, unit="vid"):
                vid_path = os.path.join(cls_dir, vid)
                npy_path = os.path.splitext(vid_path)[0] + ".npy"

                if os.path.exists(npy_path) and not force:
                    total_ok += 1
                    continue

                frames = video_to_frames(vid_path, landmarker)
                if frames is None:
                    total_fail += 1
                    continue

                frames = normalize(frames)
                np.save(npy_path, frames)
                total_ok += 1

    print(f"\nГотово: {total_ok} успешно, {total_fail} ошибок")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data",  default=DATA_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    process_dataset(args.data, force=args.force)