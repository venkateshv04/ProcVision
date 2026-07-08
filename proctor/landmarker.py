"""Shared MediaPipe Face Landmarker wrapper.

One model produces everything downstream modules need:
478 facial landmarks (incl. iris) and a facial transformation matrix
per detected face. The .task weights (~4 MB) are auto-downloaded on
first run and cached under models/.
"""

from __future__ import annotations

import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/1/face_landmarker.task"
)
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "face_landmarker.task"


def _ensure_model() -> Path:
    if not MODEL_PATH.exists():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading face landmarker model to {MODEL_PATH} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded.")
    return MODEL_PATH


class Landmarker:
    """Runs the Face Landmarker in VIDEO mode on BGR frames."""

    def __init__(self, max_faces: int = 2):
        base_options = mp_python.BaseOptions(model_asset_path=str(_ensure_model()))
        options = mp_vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_faces=max_faces,
            output_facial_transformation_matrixes=True,
            min_face_detection_confidence=0.5,
            min_face_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.FaceLandmarker.create_from_options(options)
        self._t0 = time.monotonic()
        self._last_ts_ms = -1

    def detect(self, frame_bgr: np.ndarray) -> mp_vision.FaceLandmarkerResult:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts_ms = int((time.monotonic() - self._t0) * 1000)
        # VIDEO mode requires strictly increasing timestamps
        if ts_ms <= self._last_ts_ms:
            ts_ms = self._last_ts_ms + 1
        self._last_ts_ms = ts_ms
        return self._landmarker.detect_for_video(mp_image, ts_ms)

    def close(self) -> None:
        self._landmarker.close()


def to_px(landmark, width: int, height: int) -> tuple[int, int]:
    """Convert a normalized landmark to pixel coordinates."""
    return int(landmark.x * width), int(landmark.y * height)


def open_camera(index: int = 0) -> cv2.VideoCapture:
    """Open the webcam (DirectShow backend on Windows for faster startup)."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Is another app using it?")
    return cap
