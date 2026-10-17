"""MediaPipe hand tracking. Returns the 21 landmarks of each visible hand in pixel coordinates.

Landmark indices: 0 wrist · 4 thumb tip · 8 index tip · 12 middle tip · 16 ring tip · 20 pinky tip.
"""

import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
DEFAULT_MODEL = Path("models/hand_landmarker.task")

FINGER_TIPS = {"index": 8, "middle": 12, "ring": 16, "pinky": 20}
FINGER_NUMBERS = {"index": 1, "middle": 2, "ring": 3, "pinky": 4}  # how guitar teachers number them


@dataclass
class Hand:
    landmarks_px: np.ndarray  # (21, 3): x, y in pixels; z relative depth (MediaPipe units)
    label: str  # MediaPipe's handedness label for the raw camera image
    score: float

    def tip(self, finger: str) -> np.ndarray:
        return self.landmarks_px[FINGER_TIPS[finger], :2]


def ensure_model(path: Path = DEFAULT_MODEL) -> Path:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading hand model → {path}")
        urllib.request.urlretrieve(MODEL_URL, path)
    return path


class HandTracker:
    def __init__(self, model_path: Path = DEFAULT_MODEL, max_hands: int = 2):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[Hand]:
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        h, w = frame_bgr.shape[:2]
        hands = []
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            pts = np.array([[p.x * w, p.y * h, p.z * w] for p in lms], dtype=np.float32)
            hands.append(Hand(pts, handed[0].category_name, handed[0].score))
        return hands

    def close(self) -> None:
        self._landmarker.close()
