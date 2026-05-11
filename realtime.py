"""
lipreader/realtime.py
Чтение по губам в реальном времени.
Запуск:
    python realtime.py
    python realtime.py --cam 1
"""

import os, sys, collections, time
import argparse
import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODELS_DIR, IDX_TO_CLASS, SEQUENCE_LENGTH
from src.preprocess import _make_landmarker, _crop_lip, normalize
from src.model import LipReader
import mediapipe as mp

# ── Единственные параметры которые стоит трогать ─────────────────
STEP_FRAMES  = 8     # как часто делать предсказание (в кадрах)
MIN_CONF     = 0.40  # минимальная уверенность чтобы показать слово
HOLD_SEC     = 2.0   # сколько секунд держать слово на экране/в терминале


def run(weights_path: str, cam_index: int = 0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = LipReader().to(device)
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    print(f"Готово. Устройство: {device}")
    print("Нажми Q для выхода.\n")

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        print(f"❌ Камера {cam_index} недоступна"); sys.exit(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    frame_buffer      = collections.deque(maxlen=SEQUENCE_LENGTH)
    frames_since_pred = 0

    # Текущее отображаемое слово
    current_word = ""
    current_conf = 0.0
    word_shown_at = 0.0   # когда слово появилось

    with _make_landmarker() as landmarker:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            frame_bgr = cv2.flip(frame_bgr, 1)

            # Детекция губ
            rgb      = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            result   = landmarker.detect(mp_image)

            lip_ok = False
            if result.face_landmarks:
                lip = _crop_lip(frame_bgr, result.face_landmarks[0])
                if lip is not None:
                    frame_buffer.append(lip.astype(np.float32))
                    lip_ok = True

            frames_since_pred += 1

            # Делаем предсказание
            if lip_ok and len(frame_buffer) == SEQUENCE_LENGTH and frames_since_pred >= STEP_FRAMES:
                frames_since_pred = 0

                frames = normalize(np.stack(list(frame_buffer), axis=0))
                tensor = torch.from_numpy(frames).unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    probs = model.predict_proba(tensor)[0].cpu().numpy()

                idx  = int(np.argmax(probs))
                conf = float(probs[idx])
                word = IDX_TO_CLASS[idx]

                # Показываем слово только если уверенность достаточная
                if conf >= MIN_CONF:
                    # Новое слово — выводим в терминал
                    if word != current_word:
                        print(f"  👄  {word}  ({conf:.0%})")
                        current_word  = word
                        current_conf  = conf
                        word_shown_at = time.time()
                    else:
                        # То же слово — просто обновляем уверенность
                        current_conf  = conf
                        word_shown_at = time.time()

            # Гасим слово если прошло HOLD_SEC без подтверждения
            if current_word and (time.time() - word_shown_at) > HOLD_SEC:
                current_word = ""
                current_conf = 0.0

            # Рисуем на кадре
            _draw(frame_bgr, current_word, current_conf, lip_ok, len(frame_buffer))
            cv2.imshow("LipReader  (Q — выход)", frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


def _draw(frame, word, conf, lip_ok, buf_len):
    h, w = frame.shape[:2]

    # Статус
    cv2.putText(frame, "● губы" if lip_ok else "○ губы",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                (0, 220, 0) if lip_ok else (0, 0, 200), 2)

    # Прогресс-бар буфера
    bw = int(buf_len / SEQUENCE_LENGTH * (w - 20))
    cv2.rectangle(frame, (10, 40), (w - 10, 50), (40, 40, 40), -1)
    cv2.rectangle(frame, (10, 40), (10 + bw, 50), (0, 180, 80), -1)

    # Слово крупно внизу
    if word:
        cv2.putText(frame, f"{word}  {conf:.0%}",
                    (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX,
                    1.4, (0, 255, 0), 3)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=os.path.join(MODELS_DIR, "lipreader_final.pt"))
    parser.add_argument("--cam", type=int, default=0)
    args = parser.parse_args()
    if not os.path.exists(args.weights):
        print(f"Веса не найдены: {args.weights}"); sys.exit(1)
    run(args.weights, args.cam)