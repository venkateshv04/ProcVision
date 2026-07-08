"""Concentration index — weighted fusion of the three behavioral signals.

Preserves the original project's concept: gaze, drowsiness and head
orientation each contribute a weighted share; the displayed value is
EMA-smoothed so it moves like a gauge, not a strobe.
"""

from __future__ import annotations

from .blink import EyeState
from .config import CONFIG
from .events import EMA
from .gaze import GazeDirection
from .head_pose import HeadDirection


class ConcentrationScorer:
    def __init__(self):
        self._ema = EMA(CONFIG.score.ema_alpha)
        self.raw: float = 0.0
        self.smoothed: float = 0.0

    def update(
        self,
        gaze: GazeDirection | None,
        eye_state: EyeState | None,
        head: HeadDirection | None,
        face_present: bool,
    ) -> float:
        cfg = CONFIG.score

        if not face_present:
            self.raw = 0.0
            self.smoothed = self._ema.update(0.0)
            return self.smoothed

        # Gaze is measured relative to the FACE, not the screen. With the
        # head turned, eyes counter-rotate to stay on screen — head left
        # + gaze right (or vice versa) means attention is still on the
        # screen. Same for vertical: head up + gaze down, etc.
        compensating = (
            (head == HeadDirection.LEFT and gaze == GazeDirection.RIGHT)
            or (head == HeadDirection.RIGHT and gaze == GazeDirection.LEFT)
            or (head == HeadDirection.UP and gaze == GazeDirection.DOWN)
            or (head == HeadDirection.DOWN and gaze == GazeDirection.UP)
        )
        eyes_on_screen = gaze == GazeDirection.CENTER or compensating

        gaze_score = cfg.gaze_weight if eyes_on_screen else 0.0

        # Eye-closure readings are only trustworthy near-frontal: at high
        # yaw the landmarks foreshorten, and a deep look-down narrows the
        # lids into the "closed" EAR band.
        eyes_reliable = (
            head not in (HeadDirection.LEFT, HeadDirection.RIGHT)
            and gaze != GazeDirection.DOWN
        )

        if not eyes_reliable or eye_state == EyeState.OPEN:
            drowsy_score = cfg.drowsiness_weight
        elif eye_state == EyeState.DROWSY:
            drowsy_score = cfg.drowsiness_weight * 0.5
        else:  # CLOSED or unknown
            drowsy_score = 0.0

        if head == HeadDirection.FORWARD:
            orientation_score = cfg.orientation_weight
        elif eyes_on_screen:
            # Posture deviates but attention is on the screen — mostly fine.
            orientation_score = cfg.orientation_weight * 0.75
        elif head in (HeadDirection.UP, HeadDirection.DOWN):
            orientation_score = cfg.orientation_weight * 0.5
        else:  # LEFT / RIGHT without compensation
            orientation_score = 0.0

        # Sleeping overrides everything — but only when the eye reading
        # is reliable (near-frontal head).
        if eyes_reliable and eye_state == EyeState.CLOSED:
            self.raw = 0.0
        else:
            self.raw = (gaze_score + drowsy_score + orientation_score) * 100.0

        self.smoothed = self._ema.update(self.raw)
        return self.smoothed
