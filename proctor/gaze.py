"""Gaze direction from MediaPipe iris landmarks.

The iris center position is measured relative to the eye corners
(horizontal) and eyelids (vertical), giving ratios in [0, 1] where
0.5 is centered. Both eyes are averaged.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import CONFIG
from .landmarker import to_px

# MediaPipe FaceMesh indices (refined landmarks)
RIGHT_IRIS_CENTER = 468   # subject's right eye (image left when mirrored)
LEFT_IRIS_CENTER = 473
RIGHT_EYE_OUTER, RIGHT_EYE_INNER = 33, 133
LEFT_EYE_INNER, LEFT_EYE_OUTER = 362, 263
RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM = 159, 145
LEFT_EYE_TOP, LEFT_EYE_BOTTOM = 386, 374


class GazeDirection(Enum):
    CENTER = "looking center"
    LEFT = "looking left"
    RIGHT = "looking right"
    UP = "looking up"
    DOWN = "looking down"


@dataclass
class GazeReading:
    direction: GazeDirection
    h_ratio: float
    v_ratio: float          # raw vertical offset from the eye-corner line
    right_iris_px: tuple[int, int]
    left_iris_px: tuple[int, int]
    calibrating: bool = False
    v_baseline: float | None = None
    openness: float = 0.0   # lid gap / eye width, averaged over both eyes


def _ratio(value: float, low: float, high: float) -> float:
    span = high - low
    if span == 0:
        return 0.5
    return (value - low) / span


def read_gaze(landmarks, frame_w: int, frame_h: int) -> GazeReading:
    lm = landmarks

    # Horizontal: iris x between the two eye corners
    r_h = _ratio(lm[RIGHT_IRIS_CENTER].x, lm[RIGHT_EYE_OUTER].x, lm[RIGHT_EYE_INNER].x)
    l_h = _ratio(lm[LEFT_IRIS_CENTER].x, lm[LEFT_EYE_INNER].x, lm[LEFT_EYE_OUTER].x)
    h_ratio = (r_h + l_h) / 2.0

    # Vertical: iris y offset from the eye-corner line, normalized by eye
    # width. The corner line does not move with the eyeball (unlike the
    # eyelids, which track vertical gaze and cancel the signal).
    def _v_offset(iris_i: int, corner_a: int, corner_b: int) -> float:
        corners_mid_y = (lm[corner_a].y + lm[corner_b].y) / 2.0
        eye_width = abs(lm[corner_b].x - lm[corner_a].x)
        if eye_width == 0:
            return 0.0
        return (lm[iris_i].y - corners_mid_y) / eye_width

    r_v = _v_offset(RIGHT_IRIS_CENTER, RIGHT_EYE_OUTER, RIGHT_EYE_INNER)
    l_v = _v_offset(LEFT_IRIS_CENTER, LEFT_EYE_INNER, LEFT_EYE_OUTER)
    v_ratio = (r_v + l_v) / 2.0

    # Eye openness: lid gap normalized by eye width (per eye, averaged)
    def _openness(top: int, bottom: int, corner_a: int, corner_b: int) -> float:
        eye_width = abs(lm[corner_b].x - lm[corner_a].x)
        if eye_width == 0:
            return 0.0
        return abs(lm[bottom].y - lm[top].y) / eye_width

    openness = (
        _openness(RIGHT_EYE_TOP, RIGHT_EYE_BOTTOM, RIGHT_EYE_OUTER, RIGHT_EYE_INNER)
        + _openness(LEFT_EYE_TOP, LEFT_EYE_BOTTOM, LEFT_EYE_INNER, LEFT_EYE_OUTER)
    ) / 2.0

    return GazeReading(
        direction=GazeDirection.CENTER,  # classified by GazeTracker
        h_ratio=h_ratio,
        v_ratio=v_ratio,
        right_iris_px=to_px(lm[RIGHT_IRIS_CENTER], frame_w, frame_h),
        left_iris_px=to_px(lm[LEFT_IRIS_CENTER], frame_w, frame_h),
        openness=openness,
    )


class GazeTracker:
    """Stateful gaze classifier with per-user vertical auto-calibration.

    The neutral iris-to-corner-line offset differs by anatomy and camera
    height, so absolute vertical thresholds misfire. For the first
    ~1.5 seconds (while the user naturally looks at the screen) the
    tracker collects the median offset as a baseline; afterwards,
    up/down is judged by deviation from that baseline.
    """

    def __init__(self):
        self._v_samples: list[float] = []
        self._o_samples: list[float] = []
        self._h_samples: list[float] = []
        self.v_baseline: float | None = None
        self.openness_baseline: float | None = None
        self.h_baseline: float | None = None
        self._down_since: float | None = None
        self._up_since: float | None = None
        self._guard_since: float | None = None
        self._last_horizontal: GazeDirection = GazeDirection.CENTER
        # EMA-smoothed signals
        self._h_s: float | None = None
        self._v_s: float | None = None
        self._o_s: float | None = None

    def _smooth(self, attr: str, x: float) -> float:
        alpha = CONFIG.gaze.smoothing_alpha
        prev = getattr(self, attr)
        value = x if prev is None else alpha * x + (1 - alpha) * prev
        setattr(self, attr, value)
        return value

    def _classify_horizontal(self, h: float) -> GazeDirection:
        """Deviation from the calibrated baseline, with hysteresis."""
        cfg = CONFIG.gaze
        base = self.h_baseline if self.h_baseline is not None else 0.5
        dh = h - base
        m = cfg.h_hysteresis
        if self._last_horizontal == GazeDirection.LEFT:
            direction = GazeDirection.LEFT if dh < cfg.h_left_delta + m else GazeDirection.CENTER
        elif self._last_horizontal == GazeDirection.RIGHT:
            direction = GazeDirection.RIGHT if dh > cfg.h_right_delta - m else GazeDirection.CENTER
        elif dh < cfg.h_left_delta:
            direction = GazeDirection.LEFT
        elif dh > cfg.h_right_delta:
            direction = GazeDirection.RIGHT
        else:
            direction = GazeDirection.CENTER
        self._last_horizontal = direction
        return direction

    @staticmethod
    def _median(samples: list[float]) -> float:
        s = sorted(samples)
        return s[len(s) // 2]

    def read(self, landmarks, frame_w: int, frame_h: int, now: float = 0.0) -> GazeReading:
        reading = read_gaze(landmarks, frame_w, frame_h)
        cfg = CONFIG.gaze

        if self.v_baseline is None:
            self._v_samples.append(reading.v_ratio)
            self._o_samples.append(reading.openness)
            self._h_samples.append(reading.h_ratio)
            if len(self._v_samples) >= cfg.calibration_frames:
                self.v_baseline = self._median(self._v_samples)
                self.openness_baseline = self._median(self._o_samples)
                self.h_baseline = self._median(self._h_samples)
            reading.calibrating = True
            return reading

        reading.v_baseline = self.v_baseline

        # Smooth the raw per-frame signals before classifying
        h = self._smooth("_h_s", reading.h_ratio)
        v = self._smooth("_v_s", reading.v_ratio)
        o = self._smooth("_o_s", reading.openness)
        dv = v - self.v_baseline
        openness_ratio = o / self.openness_baseline if self.openness_baseline else 1.0

        if openness_ratio < cfg.blink_guard_ratio:
            # Eyes nearly closed. A blink is brief — but if they STAY this
            # narrow with the iris low, it's a deep look-down (phone in
            # lap), not a blink.
            if self._guard_since is None:
                self._guard_since = now
            if now - self._guard_since >= cfg.guard_down_s and dv > 0:
                candidate = GazeDirection.DOWN
            else:
                candidate = GazeDirection.CENTER
        else:
            self._guard_since = None
            if dv < cfg.up_delta:
                candidate = GazeDirection.UP
            elif dv > cfg.down_delta and openness_ratio < cfg.down_openness_ratio:
                # Looking down = iris drops a little AND the top lid
                # follows, narrowing the eye.
                candidate = GazeDirection.DOWN
            else:
                candidate = self._classify_horizontal(h)

        # Debounce vertical directions: a blink's closing/opening
        # transition briefly mimics down (and sometimes up). Only report
        # them once they persist longer than any blink transition.
        if candidate == GazeDirection.DOWN:
            self._up_since = None
            if self._down_since is None:
                self._down_since = now
            # The guard path already waited guard_down_s — don't stack
            # a second debounce on top of it.
            if self._guard_since is None and now - self._down_since < cfg.vertical_debounce_s:
                candidate = GazeDirection.CENTER
        elif candidate == GazeDirection.UP:
            self._down_since = None
            if self._up_since is None:
                self._up_since = now
            if now - self._up_since < cfg.vertical_debounce_s:
                candidate = GazeDirection.CENTER
        else:
            self._down_since = None
            self._up_since = None

        reading.direction = candidate
        return reading
