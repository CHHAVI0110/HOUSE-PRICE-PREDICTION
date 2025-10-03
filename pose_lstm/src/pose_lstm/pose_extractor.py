from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple, List

try:
    import mediapipe as mp
except ImportError as exc:
    raise RuntimeError(
        "mediapipe is required. Please install with `pip install mediapipe`."
    ) from exc


# MediaPipe landmark indices for convenience
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12
LEFT_HIP = 23
RIGHT_HIP = 24
NUM_LANDMARKS = 33


@dataclass
class PoseExtractorConfig:
    static_image_mode: bool = False
    model_complexity: int = 1
    enable_segmentation: bool = False
    smooth_landmarks: bool = True
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    include_visibility: bool = True
    normalize: bool = True
    normalize_center: str = "mid_hip"  # options: mid_hip, mid_shoulder, nose
    normalize_scale: str = "shoulder"   # options: shoulder, hip, torso
    add_angles: bool = False            # if True, append a few joint angles


class PoseExtractor:
    """Wrapper around MediaPipe Pose for extracting and normalizing landmarks."""

    def __init__(self, config: PoseExtractorConfig | None = None) -> None:
        self.config = config or PoseExtractorConfig()
        self._mp_pose = mp.solutions.pose
        self._pose = self._mp_pose.Pose(
            static_image_mode=self.config.static_image_mode,
            model_complexity=self.config.model_complexity,
            enable_segmentation=self.config.enable_segmentation,
            smooth_landmarks=self.config.smooth_landmarks,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )

    def close(self) -> None:
        self._pose.close()

    def __enter__(self) -> "PoseExtractor":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @staticmethod
    def _get_midpoint(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return (a + b) / 2.0

    @staticmethod
    def _euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.linalg.norm(a - b))

    def _compute_scale(self, pts: np.ndarray) -> float:
        left_shoulder = pts[LEFT_SHOULDER, :2]
        right_shoulder = pts[RIGHT_SHOULDER, :2]
        left_hip = pts[LEFT_HIP, :2]
        right_hip = pts[RIGHT_HIP, :2]

        if self.config.normalize_scale == "shoulder":
            scale = self._euclidean_distance(left_shoulder, right_shoulder)
        elif self.config.normalize_scale == "hip":
            scale = self._euclidean_distance(left_hip, right_hip)
        else:  # torso
            shoulder_mid = self._get_midpoint(left_shoulder, right_shoulder)
            hip_mid = self._get_midpoint(left_hip, right_hip)
            scale = self._euclidean_distance(shoulder_mid, hip_mid)

        if scale < 1e-5:
            # Fallbacks to avoid division by zero in rare cases
            scale = (
                self._euclidean_distance(left_hip, right_hip)
                or self._euclidean_distance(left_shoulder, right_shoulder)
                or 1.0
            )
        return scale

    def _select_center(self, pts: np.ndarray) -> np.ndarray:
        left_shoulder = pts[LEFT_SHOULDER, :2]
        right_shoulder = pts[RIGHT_SHOULDER, :2]
        left_hip = pts[LEFT_HIP, :2]
        right_hip = pts[RIGHT_HIP, :2]

        if self.config.normalize_center == "mid_shoulder":
            center = self._get_midpoint(left_shoulder, right_shoulder)
        else:  # mid_hip (default) or nose
            if self.config.normalize_center == "nose":
                center = pts[0, :2]
            else:
                center = self._get_midpoint(left_hip, right_hip)
        return center

    def _append_joint_angles(self, pts_xyz: np.ndarray) -> np.ndarray:
        """Optionally compute a few angles and append as features.

        Angles computed (in degrees):
        - Left and right elbow flexion
        - Left and right knee flexion
        """
        # Define helper to compute angle at b formed by a-b-c
        def angle(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
            ba = a - b
            bc = c - b
            denom = (np.linalg.norm(ba) * np.linalg.norm(bc))
            if denom < 1e-8:
                return 0.0
            cosang = float(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))
            return float(np.degrees(np.arccos(cosang)))

        # MediaPipe landmark indices
        LEFT_ELBOW, RIGHT_ELBOW = 13, 14
        LEFT_WRIST, RIGHT_WRIST = 15, 16
        LEFT_KNEE, RIGHT_KNEE = 25, 26
        LEFT_ANKLE, RIGHT_ANKLE = 27, 28

        angles: List[float] = []
        # Elbow angles
        angles.append(angle(pts_xyz[LEFT_SHOULDER], pts_xyz[LEFT_ELBOW], pts_xyz[LEFT_WRIST]))
        angles.append(angle(pts_xyz[RIGHT_SHOULDER], pts_xyz[RIGHT_ELBOW], pts_xyz[RIGHT_WRIST]))
        # Knee angles
        angles.append(angle(pts_xyz[LEFT_HIP], pts_xyz[LEFT_KNEE], pts_xyz[LEFT_ANKLE]))
        angles.append(angle(pts_xyz[RIGHT_HIP], pts_xyz[RIGHT_KNEE], pts_xyz[RIGHT_ANKLE]))

        return np.concatenate([pts_xyz.flatten(), np.asarray(angles, dtype=np.float32)], axis=0)

    def landmarks_from_bgr(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Run MediaPipe Pose on a BGR image and return Nx4 array (x,y,z,vis).

        Returns None if pose not detected.
        """
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = self._pose.process(image_rgb)
        if not results.pose_landmarks:
            return None

        lm = results.pose_landmarks.landmark
        landmarks = np.zeros((NUM_LANDMARKS, 4), dtype=np.float32)
        for i in range(NUM_LANDMARKS):
            l = lm[i]
            landmarks[i, 0] = l.x
            landmarks[i, 1] = l.y
            landmarks[i, 2] = l.z
            landmarks[i, 3] = getattr(l, "visibility", 1.0)
        return landmarks

    def normalize_landmarks(self, landmarks: np.ndarray) -> np.ndarray:
        """Normalize landmarks for scale and translation. Returns features as (N, F)."""
        # Separate spatial and visibility
        xy = landmarks[:, :2].copy()
        z = landmarks[:, 2:3].copy()
        vis = landmarks[:, 3:4].copy() if self.config.include_visibility else None

        center = self._select_center(landmarks)
        xy -= center

        # Use image-relative coordinates; scale normalization via body anchors
        scale = self._compute_scale(landmarks)
        xy /= scale
        z /= scale

        pts_xyz = np.concatenate([xy, z], axis=1)  # (33, 3)

        if self.config.add_angles:
            feat = self._append_joint_angles(pts_xyz)
            # Optionally include visibility summary statistics
            if vis is not None:
                vis_stats = np.concatenate([
                    vis.mean(axis=0), vis.min(axis=0), vis.max(axis=0)
                ], axis=0).astype(np.float32)
                feat = np.concatenate([feat, vis_stats], axis=0)
            return feat

        # Flatten landmarks into 1D feature vector
        if vis is not None:
            feat = np.concatenate([pts_xyz, vis], axis=1).astype(np.float32)
        else:
            feat = pts_xyz.astype(np.float32)
        return feat.flatten()

    def features_from_bgr(self, image_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Convenience: run inference then normalize+flatten into feature vector."""
        lm = self.landmarks_from_bgr(image_bgr)
        if lm is None:
            return None
        return self.normalize_landmarks(lm)


def draw_pose(image_bgr: np.ndarray, pose_result) -> np.ndarray:
    """Draw landmarks using MediaPipe utilities if pose_result available."""
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    image_out = image_bgr.copy()
    if pose_result and pose_result.pose_landmarks:
        mp_drawing.draw_landmarks(
            image=image_out,
            landmark_list=pose_result.pose_landmarks,
            connections=mp.solutions.pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
        )
    return image_out
