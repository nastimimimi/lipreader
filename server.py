"""
lipreader/server.py
Flask сервер с поддержкой сбора датасета через голосование.
Запуск:
    pip install flask
    python server.py
"""

import os, sys, tempfile, uuid, json, shutil
from flask import Flask, request, jsonify, send_from_directory
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODELS_DIR, IDX_TO_CLASS, DATA_DIR, CLASSES
from src.preprocess import _make_landmarker, video_to_frames, normalize
from src.model import LipReader

app = Flask(__name__, static_folder="web")

# Папка для временных видео ожидающих разметки
PENDING_DIR = os.path.join(os.path.dirname(__file__), "pending_videos")
os.makedirs(PENDING_DIR, exist_ok=True)

# ── Загружаем модель один раз ────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = LipReader().to(device)
state  = torch.load(os.path.join(MODELS_DIR, "lipreader_final.pt"), map_location=device)
if isinstance(state, dict) and "model_state" in state:
    state = state["model_state"]
model.load_state_dict(state)
model.eval()
print(f"Модель загружена | устройство: {device}")


@app.route("/")
def index():
    return send_from_directory("web", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "video" not in request.files:
        return jsonify({"error": "Видео не получено"}), 400

    video_file = request.files["video"]
    ext = "mp4" if "mp4" in video_file.content_type else "webm"

    # Уникальный ID для этого видео
    video_id = str(uuid.uuid4())
    tmp_path  = os.path.join(PENDING_DIR, f"{video_id}.{ext}")
    video_file.save(tmp_path)

    try:
        with _make_landmarker() as landmarker:
            frames = video_to_frames(tmp_path, landmarker)

        if frames is None:
            os.unlink(tmp_path)
            return jsonify({"error": "Губы не найдены в видео"}), 422

        frames = normalize(frames)
        tensor = torch.from_numpy(frames).unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            probs = model.predict_proba(tensor)[0].cpu().numpy()

        top3 = np.argsort(probs)[::-1][:3]
        results = [
            {"word": IDX_TO_CLASS[i], "confidence": round(float(probs[i]), 3)}
            for i in top3
        ]

        # Сохраняем метаданные рядом с видео
        meta = {"predicted": results[0]["word"], "ext": ext}
        with open(os.path.join(PENDING_DIR, f"{video_id}.json"), "w") as f:
            json.dump(meta, f)

        return jsonify({"results": results, "video_id": video_id})

    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        return jsonify({"error": str(e)}), 500


@app.route("/label", methods=["POST"])
def label():
    """Принимает разметку пользователя и перекладывает видео в датасет."""
    data     = request.get_json()
    video_id = data.get("video_id")
    label    = data.get("label")

    if not video_id or not label:
        return jsonify({"ok": False, "error": "Нет video_id или label"}), 400

    if label not in CLASSES:
        return jsonify({"ok": False, "error": f"Неизвестный класс: {label}"}), 400

    # Ищем видео в pending
    meta_path = os.path.join(PENDING_DIR, f"{video_id}.json")
    if not os.path.exists(meta_path):
        return jsonify({"ok": False, "error": "Видео не найдено"}), 404

    with open(meta_path) as f:
        meta = json.load(f)

    ext      = meta.get("ext", "mp4")
    vid_path = os.path.join(PENDING_DIR, f"{video_id}.{ext}")

    if not os.path.exists(vid_path):
        return jsonify({"ok": False, "error": "Файл видео не найден"}), 404

    # Перекладываем в data/<label>/
    dest_dir = os.path.join(DATA_DIR, label)
    os.makedirs(dest_dir, exist_ok=True)
    dest_path = os.path.join(dest_dir, f"web_{video_id}.{ext}")
    shutil.move(vid_path, dest_path)

    # Удаляем метафайл
    os.unlink(meta_path)

    print(f"  ✓ Сохранено: {label}/web_{video_id}.{ext}")
    return jsonify({"ok": True})


@app.route("/stats")
def stats():
    """Возвращает количество размеченных видео по классам."""
    counts = {}
    for cls in CLASSES:
        cls_dir = os.path.join(DATA_DIR, cls)
        if os.path.isdir(cls_dir):
            # Только оригинальные видео (без аугментации и .npy)
            n = len([f for f in os.listdir(cls_dir)
                     if f.endswith(('.mp4', '.webm', '.mov'))
                     and '_aug_' not in f])
            counts[cls] = n
        else:
            counts[cls] = 0
    return jsonify({"counts": counts})


if __name__ == "__main__":
    app.run(debug=False, port=5000)