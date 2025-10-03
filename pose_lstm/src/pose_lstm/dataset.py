from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict

import numpy as np
import cv2
import pandas as pd
from torch.utils.data import Dataset

from .pose_extractor import PoseExtractor, PoseExtractorConfig


@dataclass
class SequenceConfig:
    sequence_length: int = 30
    step: int = 1
    drop_incomplete: bool = True


class PoseSequenceDataset(Dataset):
    """Loads pre-extracted per-frame features from .npy files and yields sequences.

    Directory layout:
    data/
      class_name_1/
        video1.npy  # shape (T, F)
        video2.npy
      class_name_2/
        video3.npy

    Labels are derived from folder names.
    """

    def __init__(self, root_dir: str, config: SequenceConfig | None = None) -> None:
        self.root_dir = root_dir
        self.config = config or SequenceConfig()
        self.samples: List[Tuple[str, int]] = []  # (path, label)
        self.class_to_idx: Dict[str, int] = {}
        self._scan()

    def _scan(self) -> None:
        classes = sorted([d for d in os.listdir(self.root_dir) if os.path.isdir(os.path.join(self.root_dir, d))])
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        for c in classes:
            cdir = os.path.join(self.root_dir, c)
            for fname in sorted(os.listdir(cdir)):
                if not fname.endswith(".npy"):
                    continue
                path = os.path.join(cdir, fname)
                self.samples.append((path, self.class_to_idx[c]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        arr = np.load(path)  # shape (T, F)
        T = arr.shape[0]
        L = self.config.sequence_length
        S = self.config.step

        # Build sequences for this sample; return one randomly for training-like behavior
        start_indices = list(range(0, max(1, T - L + 1), S))
        if self.config.drop_incomplete:
            start_indices = [s for s in start_indices if s + L <= T]

        if not start_indices:
            # Pad or repeat last frame if needed
            if T == 0:
                seq = np.zeros((L, arr.shape[1]), dtype=np.float32)
            else:
                pad_len = L - T
                pad = np.repeat(arr[-1:, :], pad_len, axis=0)
                seq = np.concatenate([arr, pad], axis=0)
        else:
            s = np.random.choice(start_indices)
            seq = arr[s : s + L]

        return seq.astype(np.float32), int(label)


def extract_features_from_video(video_path: str, extractor: PoseExtractor) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    feats: List[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        feat = extractor.features_from_bgr(frame)
        if feat is not None:
            feats.append(feat)
    cap.release()

    if not feats:
        return None
    return np.stack(feats, axis=0).astype(np.float32)


def bulk_extract_dataset(input_root: str, output_root: str, extractor: PoseExtractor) -> None:
    os.makedirs(output_root, exist_ok=True)
    for class_name in sorted(os.listdir(input_root)):
        in_dir = os.path.join(input_root, class_name)
        if not os.path.isdir(in_dir):
            continue
        out_dir = os.path.join(output_root, class_name)
        os.makedirs(out_dir, exist_ok=True)

        for fname in sorted(os.listdir(in_dir)):
            if not (fname.endswith('.mp4') or fname.endswith('.avi') or fname.endswith('.mov')):
                continue
            in_path = os.path.join(in_dir, fname)
            out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + '.npy')
            if os.path.exists(out_path):
                continue
            feats = extract_features_from_video(in_path, extractor)
            if feats is not None:
                np.save(out_path, feats)
