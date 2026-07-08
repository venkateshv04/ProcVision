"""Blink and drowsiness signal via the Eye Aspect Ratio (EAR).

EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|) per eye, using the six-point
eye contour. Sustained low EAR indicates drowsiness or sleep; the
temporal judgement lives in events.py, this module reports per-frame state.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .config import CONFIG
from .landmarker import to_px

# Six-point eye contours (MediaPipe FaceMesh indices), order p1..p6
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_EYE = [362, 385, 387, 263, 373, 380]


class EyeState(Enum):
    OPEN = "eyes open"
    DROWSY = "drowsy"
    CLOSED = "eyes closed"


@dataclass
class BlinkReading:
    state: EyeState
    ear_right: float
    ear_left: float
    ear_avg: float
    right_eye_px: list[tuple[int, int]]
    left_eye_px: list[tuple[int, int]]


def _dist(a: tuple[int, int], b: tuple[int, int]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _ear(points: list[tuple[int, int]]) -> float:
    p1, p2, p3, p4, p5, p6 = points
    horizontal = _dist(p1, p4)
    if horizontal == 0:
        return 0.0
    return (_dist(p2, p6) + _dist(p3, p5)) / (2.0 * horizontal)


def read_blink(landmarks, frame_w: int, frame_h: int) -> BlinkReading:
    right_px = [to_px(landmarks[i], frame_w, frame_h) for i in RIGHT_EYE]
    left_px = [to_px(landmarks[i], frame_w, frame_h) for i in LEFT_EYE]

    ear_r = _ear(right_px)
    ear_l = _ear(left_px)
    ear_avg = (ear_r + ear_l) / 2.0

    cfg = CONFIG.blink
    if ear_avg < cfg.closed_threshold:
        state = EyeState.CLOSED
    elif ear_avg < cfg.drowsy_threshold:
        state = EyeState.DROWSY
    else:
        state = EyeState.OPEN

    return BlinkReading(
        state=state,
        ear_right=ear_r,
        ear_left=ear_l,
        ear_avg=ear_avg,
        right_eye_px=right_px,
        left_eye_px=left_px,
    )


class BlinkTracker:
    """Time-debounced eye state.

    Humans blink every few seconds; a ~0.2 s closure is normal and must
    not read as 'eyes closed'. Raw per-frame state only becomes CLOSED
    or DROWSY after persisting past the configured durations.
    """

    def __init__(self):
        self._closed_since: float | None = None
        self._drowsy_since: float | None = None

    def update(self, reading: BlinkReading, now: float) -> EyeState:
        cfg = CONFIG.blink
        raw = reading.state

        if raw == EyeState.CLOSED:
            self._drowsy_since = None
            if self._closed_since is None:
                self._closed_since = now
            if now - self._closed_since >= cfg.blink_ignore_s:
                return EyeState.CLOSED
            return EyeState.OPEN  # just a blink

        self._closed_since = None

        if raw == EyeState.DROWSY:
            if self._drowsy_since is None:
                self._drowsy_since = now
            if now - self._drowsy_since >= cfg.drowsy_min_s:
                return EyeState.DROWSY
            return EyeState.OPEN

        self._drowsy_since = None
        return EyeState.OPEN
