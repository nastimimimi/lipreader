"""
lipreader/predict.py

Инференс на одном видеофайле.

Использование:
    python predict.py video.mp4
    python predict.py video.mp4 --weights models/lipreader_final.pt
    python predict.py video.mp4 --top3
"""

import os
import sys
import argparse
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import MODELS_DIR, IDX_TO_CLASS, CONFIDENCE_THRESHOLD
from src.preprocess import video_to_frames, normalize, _make_landmarker
from src.model import LipReader


def predict(video_path: str,
            weights_path: str,
            top_k: int = 1,
            verbose: bool = True):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Загружаем модель
    model = LipReader().to(device)
    state = torch.load(weights_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()

    if verbose:
        print(f"Обрабатываем: {video_path}")

    # Препроцессинг с landmarker
    with _make_landmarker() as landmarker:
        frames = video_to_frames(video_path, landmarker)

    if frames is None:
        print("❌ Не удалось извлечь губы из видео.")
        return []

    frames = normalize(frames)
    tensor = torch.from_numpy(frames).unsqueeze(0).unsqueeze(0)  # (1,1,T,H,W)
    tensor = tensor.to(device)

    with torch.no_grad():
        probs = model.predict_proba(tensor)[0].cpu().numpy()

    top_idx = np.argsort(probs)[::-1][:top_k]
    results = [(IDX_TO_CLASS[i], float(probs[i])) for i in top_idx]

    if verbose:
        print("\n── Результат ─────────────────────────")
        for rank, (cls, prob) in enumerate(results, 1):
            bar  = "█" * int(prob * 20)
            flag = " ✓" if prob >= CONFIDENCE_THRESHOLD and rank == 1 else ""
            print(f"  #{rank} {cls:<8} {prob:.3f}  {bar}{flag}")
        if results[0][1] < CONFIDENCE_THRESHOLD:
            print(f"\n  ⚠ Уверенность ниже порога ({CONFIDENCE_THRESHOLD})")
        print("─────────────────────────────────────")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Инференс LipReader")
    parser.add_argument("video", help="Путь к видеофайлу")
    parser.add_argument(
        "--weights",
        default=os.path.join(MODELS_DIR, "lipreader_final.pt"),
    )
    parser.add_argument("--top3", action="store_true")
    args = parser.parse_args()

    if not os.path.exists(args.video):
        print(f"Файл не найден: {args.video}")
        sys.exit(1)
    if not os.path.exists(args.weights):
        print(f"Веса не найдены: {args.weights}")
        sys.exit(1)

    top_k = 3 if args.top3 else 1
    predict(args.video, args.weights, top_k=top_k)