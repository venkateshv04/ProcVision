"""Head pose (yaw / pitch / roll) from the facial transformation matrix.

MediaPipe's Face Landmarker returns a 4x4 matrix mapping the canonical
face model into camera space — its rotation block gives head orientation
directly, with no solvePnP or camera-intrinsics guesswork.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from .config import CONFIG


class HeadDirection(Enum):
    FORWARD = "facing forward"
    LEFT = "facing left"
    RIGHT = "facing right"
    UP = "facing up"
    DOWN = "facing down"


@dataclass
class HeadPoseReading:
    direction: HeadDirection
    yaw: float    # degrees, + = facing left in the mirrored view
    pitch: float  # degrees, + = looking up
    roll: float   # degrees, head tilt


def _euler_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """Extract Tait-Bryan angles (degrees) from a 4x4 transformation matrix."""
    r = m[:3, :3]
    sy = math.sqrt(r[0, 0] ** 2 + r[1, 0] ** 2)
    if sy > 1e-6:
        pitch = math.atan2(r[2, 1], r[2, 2])
        yaw = math.atan2(-r[2, 0], sy)
        roll = math.atan2(r[1, 0], r[0, 0])
    else:  # gimbal lock
        pitch = math.atan2(-r[1, 2], r[1, 1])
        yaw = math.atan2(-r[2, 0], sy)
        roll = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)


def read_head_pose(transformation_matrix: np.ndarray) -> HeadPoseReading:
    pitch, yaw, roll = _euler_from_matrix(np.asarray(transformation_matrix))
    # Align signs with the mirrored (selfie) view: + yaw = facing left,
    # + pitch = looking up.
    yaw = -yaw
    pitch = -pitch

    cfg = CONFIG.head_pose
    if yaw < -cfg.yaw_threshold:
        direction = HeadDirection.RIGHT
    elif yaw > cfg.yaw_threshold:
        direction = HeadDirection.LEFT
    elif pitch > cfg.pitch_up_threshold:
        direction = HeadDirection.UP
    elif pitch < -cfg.pitch_down_threshold:
        direction = HeadDirection.DOWN
    else:
        direction = HeadDirection.FORWARD

    return HeadPoseReading(direction=direction, yaw=yaw, pitch=pitch, roll=roll)


def axis_endpoints(
    reading: HeadPoseReading, origin: tuple[int, int], length: float = 70.0
) -> dict[str, tuple[int, int]]:
    """2D projection of the head's X/Y/Z axes for visualization."""
    yaw_r = math.radians(reading.yaw)
    pitch_r = math.radians(reading.pitch)
    roll_r = math.radians(reading.roll)
    ox, oy = origin

    x_end = (
        int(ox + length * (math.cos(yaw_r) * math.cos(roll_r))),
        int(oy + length * (math.cos(pitch_r) * math.sin(roll_r))),
    )
    y_end = (
        int(ox - length * (math.cos(yaw_r) * math.sin(roll_r))),
        int(oy - length * (math.cos(pitch_r) * math.cos(roll_r))),
    )
    z_end = (
        int(ox - length * math.sin(yaw_r)),
        int(oy + length * math.sin(pitch_r)),
    )
    return {"x": x_end, "y": y_end, "z": z_end}
