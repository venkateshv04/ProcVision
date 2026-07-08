"""Temporal event layer — turns noisy per-frame readings into events.

A condition only becomes a flagged event after it persists for a
configured duration (hysteresis), so a single glance or slow blink
never triggers a flag. Events carry start/end timestamps and severity.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from .config import CONFIG


class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Event:
    kind: str
    detail: str
    severity: Severity
    t_start: float
    t_end: float | None = None  # None while still ongoing

    @property
    def duration(self) -> float:
        end = self.t_end if self.t_end is not None else self.t_start
        return end - self.t_start


class _SustainedFlag:
    """Tracks one condition; emits an Event once it persists long enough."""

    def __init__(self, kind: str, sustain_s: float, severity: Severity):
        self.kind = kind
        self.sustain_s = sustain_s
        self.severity = severity
        self._active_since: float | None = None
        self._open_event: Event | None = None

    def update(self, active: bool, now: float, detail: str = "") -> Event | None:
        """Returns a new Event the moment the sustain threshold is crossed."""
        if active:
            if self._active_since is None:
                self._active_since = now
            elif self._open_event is None and now - self._active_since >= self.sustain_s:
                self._open_event = Event(
                    kind=self.kind,
                    detail=detail or self.kind,
                    severity=self.severity,
                    t_start=self._active_since,
                )
                return self._open_event
        else:
            if self._open_event is not None:
                self._open_event.t_end = now
                self._open_event = None
            self._active_since = None
        return None

    @property
    def is_flagged(self) -> bool:
        return self._open_event is not None


@dataclass
class FrameObservation:
    """Per-frame boolean conditions fed by the main loop."""
    face_absent: bool = False
    multiple_faces: bool = False
    gaze_off: bool = False
    gaze_detail: str = ""
    head_turned: bool = False
    head_detail: str = ""
    eyes_closed: bool = False
    drowsy: bool = False


class EventDetector:
    def __init__(self):
        cfg = CONFIG.events
        self._flags = {
            "face_absent": _SustainedFlag("Face absent", cfg.face_absent_sustain, Severity.CRITICAL),
            "multiple_faces": _SustainedFlag("Multiple faces", cfg.multiple_faces_sustain, Severity.CRITICAL),
            "gaze_off": _SustainedFlag("Gaze off-screen", cfg.gaze_off_sustain, Severity.WARNING),
            "head_turned": _SustainedFlag("Head turned away", cfg.head_turned_sustain, Severity.WARNING),
            "eyes_closed": _SustainedFlag("Eyes closed", cfg.eyes_closed_sustain, Severity.WARNING),
            "drowsy": _SustainedFlag("Drowsy", cfg.drowsy_sustain, Severity.INFO),
        }
        self.events: list[Event] = []

    def update(self, obs: FrameObservation, now: float) -> list[Event]:
        """Feed one frame; returns any events that just fired."""
        checks = [
            ("face_absent", obs.face_absent, "no face in frame"),
            ("multiple_faces", obs.multiple_faces, "second face detected"),
            ("gaze_off", obs.gaze_off, obs.gaze_detail),
            ("head_turned", obs.head_turned, obs.head_detail),
            ("eyes_closed", obs.eyes_closed, "prolonged eye closure"),
            ("drowsy", obs.drowsy, "sustained low eye aspect ratio"),
        ]
        fired = []
        for name, active, detail in checks:
            event = self._flags[name].update(active, now, detail)
            if event is not None:
                self.events.append(event)
                fired.append(event)
        return fired

    @property
    def active_flags(self) -> list[str]:
        return [f.kind for f in self._flags.values() if f.is_flagged]

    def summary(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for e in self.events:
            counts[e.kind] = counts.get(e.kind, 0) + 1
        return counts


class EMA:
    """Exponential moving average for smoothing displayed values."""

    def __init__(self, alpha: float):
        self.alpha = alpha
        self._value: float | None = None

    def update(self, x: float) -> float:
        if self._value is None:
            self._value = x
        else:
            self._value = self.alpha * x + (1 - self.alpha) * self._value
        return self._value
