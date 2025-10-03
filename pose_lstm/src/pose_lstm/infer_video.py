from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Deque, Optional
from collections import deque

import cv2
import numpy as np
import torch

from pose_lstm.pose_extractor import PoseExtractor, PoseExtractorConfig
from pose_lstm.lstm_classifier import LSTMClassifier, LSTMConfig


@dataclass
class InferenceConfig:
    model_path: str = "models/best_lstm.pth"
    sequence_length: int = 30
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    threshold: float = 0.5


def load_model(model_path: str, device: str) -> tuple[LSTMClassifier, dict]:
    ckpt = torch.load(model_path, map_location=device)
    cfg = LSTMConfig(**ckpt["config"])
    model = LSTMClassifier(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    meta = {
        "input_size": ckpt.get("input_size", cfg.input_size),
        "sequence_length": ckpt.get("sequence_length", 30),
        "class_to_idx": ckpt.get("class_to_idx", {}),
    }
    return model, meta


def softmax(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=-1)


def infer_on_video(video_path: int | str, cfg: InferenceConfig) -> None:
    model, meta = load_model(cfg.model_path, cfg.device)
    seq_len = meta.get("sequence_length", cfg.sequence_length)

    pose = PoseExtractor(PoseExtractorConfig())

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    window: Deque[np.ndarray] = deque(maxlen=seq_len)

    idx_to_class = {v: k for k, v in meta.get("class_to_idx", {}).items()}
    if not idx_to_class:
        # Default to binary correct/incorrect for display
        idx_to_class = {0: "incorrect", 1: "correct"}

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        feat = pose.features_from_bgr(frame)
        if feat is not None:
            window.append(feat)

        if len(window) == seq_len:
            x = torch.tensor(np.array([list(window)], dtype=np.float32)).to(cfg.device)
            with torch.no_grad():
                logits = model(x)
                probs = softmax(logits)[0].cpu().numpy()
                pred_idx = int(np.argmax(probs))
                pred_name = idx_to_class.get(pred_idx, str(pred_idx))
                pred_conf = float(probs[pred_idx])

            label_text = f"{pred_name} ({pred_conf:.2f})"
            color = (0, 200, 0) if pred_name == "correct" else (0, 0, 200)
            cv2.putText(frame, label_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2, cv2.LINE_AA)

        cv2.imshow("LSTM Exercise Checker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    cfg = InferenceConfig()
    # 0 for webcam; or path to a video file
    infer_on_video(0, cfg)
