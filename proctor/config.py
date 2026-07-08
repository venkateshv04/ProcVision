"""Central configuration — every tunable threshold in one place.

Values were chosen for a demo-friendly feel (fast feedback). For a stricter
real-world proctoring deployment, raise the sustain durations.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GazeConfig:
    # Horizontal gaze is classified by DEVIATION from the per-user
    # baseline captured during calibration (neutral iris position is not
    # exactly 0.5 for everyone — assuming it is makes one side trigger
    # earlier than the other).
    h_left_delta: float = -0.09
    h_right_delta: float = 0.09
    # Vertical gaze is classified by DEVIATION from a per-user baseline
    # learned during the first `calibration_frames` frames (the neutral
    # iris offset varies by anatomy and camera height).
    up_delta: float = -0.05
    # Looking down drags the top eyelid with the eyeball, so the iris
    # offset alone moves little; "down" = small iris drop AND reduced
    # eye openness (but not collapsed, which is a blink). Tuned to catch
    # the phone-in-lap posture early.
    down_delta: float = 0.018
    down_openness_ratio: float = 0.90   # openness below 90% of baseline
    blink_guard_ratio: float = 0.45     # below this it's a blink, not gaze
    # ...but a blink is BRIEF. If the eyes stay that narrow beyond this
    # duration with the iris low, it's a deep look-down (phone in lap),
    # not a blink.
    guard_down_s: float = 0.35
    # Vertical directions must persist this long before being reported —
    # a blink's closing/opening transition mimics "down" for ~0.1 s.
    vertical_debounce_s: float = 0.3
    # Per-frame ratios are EMA-smoothed, and horizontal classification
    # uses hysteresis: enter a direction at the threshold, exit it only
    # `h_hysteresis` back toward center — kills boundary flicker.
    smoothing_alpha: float = 0.45
    h_hysteresis: float = 0.04
    calibration_frames: int = 45  # ~1.5s at 30 fps


@dataclass(frozen=True)
class HeadPoseConfig:
    # Degrees of deviation from frontal before a direction is reported.
    # Generous by design: people watch a screen from a wide cone of head
    # postures — gaze, not posture, is the primary attention signal.
    yaw_threshold: float = 22.0
    pitch_up_threshold: float = 28.0
    pitch_down_threshold: float = 18.0


@dataclass(frozen=True)
class BlinkConfig:
    # Eye Aspect Ratio boundaries (MediaPipe landmarks, pixel space).
    closed_threshold: float = 0.18
    drowsy_threshold: float = 0.22
    # Temporal debounce: closures shorter than this are normal blinks
    # and must not affect the score or events.
    blink_ignore_s: float = 0.4
    # Low-but-open EAR must persist this long before counting as drowsy.
    drowsy_min_s: float = 0.8


@dataclass(frozen=True)
class EventConfig:
    # Seconds a condition must persist before it becomes a flagged event.
    gaze_off_sustain: float = 1.5
    head_turned_sustain: float = 1.5
    face_absent_sustain: float = 1.0
    multiple_faces_sustain: float = 0.5
    eyes_closed_sustain: float = 2.0
    drowsy_sustain: float = 3.0


@dataclass(frozen=True)
class ScoreConfig:
    # Weighted fusion for the concentration index (class mode).
    gaze_weight: float = 0.4
    drowsiness_weight: float = 0.3
    orientation_weight: float = 0.3
    # Exponential moving average factor for the displayed score.
    ema_alpha: float = 0.15


@dataclass(frozen=True)
class Config:
    gaze: GazeConfig = field(default_factory=GazeConfig)
    head_pose: HeadPoseConfig = field(default_factory=HeadPoseConfig)
    blink: BlinkConfig = field(default_factory=BlinkConfig)
    events: EventConfig = field(default_factory=EventConfig)
    score: ScoreConfig = field(default_factory=ScoreConfig)


CONFIG = Config()
