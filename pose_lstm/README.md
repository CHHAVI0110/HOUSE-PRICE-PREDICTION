## Pose LSTM: Exercise Correctness Detection with MediaPipe + PyTorch

This project demonstrates how to use MediaPipe Pose to extract human pose keypoints and train an LSTM classifier to detect whether an exercise is performed correctly or not.

### Features
- MediaPipe Pose extraction and robust normalization (center + scale)
- Sequence dataset utilities to turn videos into feature sequences
- PyTorch LSTM sequence classifier
- Training loop with validation and checkpoints
- Real-time/webcam inference with sliding window

### Install
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you encounter issues with `mediapipe` on Linux, install build dependencies or use a manylinux-compatible wheel.

### Prepare Data
Organize raw videos in `data_raw/` with folders for each class (e.g. `correct`, `incorrect`).
```
data_raw/
  correct/
    squat_good_1.mp4
    squat_good_2.mp4
  incorrect/
    squat_bad_1.mp4
```

Extract pose features to `.npy` files:
```python
from pose_lstm.pose_extractor import PoseExtractor, PoseExtractorConfig
from pose_lstm.dataset import bulk_extract_dataset

extractor = PoseExtractor(PoseExtractorConfig())
bulk_extract_dataset("data_raw", "data_npy", extractor)
```

This will create `data_npy/<class>/<video>.npy` with arrays shaped `(T, F)` where `F` is the feature dimension. By default, features contain all 33 landmarks with normalized `(x,y,z)` and visibility → `F = 33*4 = 132`.

### Train
```bash
PYTHONPATH=src python -m pose_lstm.train
```

Adjust hyperparameters in `TrainConfig` inside `src/pose_lstm/train.py`. A best checkpoint is saved to `models/best_lstm.pth`.

### Inference (Webcam or Video)
```bash
PYTHONPATH=src python -m pose_lstm.infer_video  # uses webcam 0
# Or
PYTHONPATH=src python -m pose_lstm.infer_video /path/to/video.mp4
```

The window size for inference should match the training `sequence_length` (default 30). The overlay will show the predicted class and confidence. If you trained with `correct`/`incorrect` class folders, it will display those names.

### Tips
- Ensure examples are representative and balanced across classes
- Consider adding angle features (`add_angles=True`) for better form understanding
- Use data augmentation at feature-level (temporal jittering, scaling noise)
- For deployment, consider exporting to TorchScript and running pose + model in a single pipeline
