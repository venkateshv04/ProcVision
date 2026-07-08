"""Face presence signal — the gate for the rest of the pipeline.

Reports how many faces are visible and the bounding box of the primary
face. NO_FACE and MULTIPLE are themselves proctoring signals.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class FaceStatus(Enum):
    NO_FACE = "no face"
    OK = "1 face"
    MULTIPLE = "multiple faces"


@dataclass
class FaceReading:
    status: FaceStatus
    count: int
    bbox: tuple[int, int, int, int] | None  # x1, y1, x2, y2 of primary face


def read_face(result, frame_w: int, frame_h: int) -> FaceReading:
    faces = result.face_landmarks
    if not faces:
        return FaceReading(FaceStatus.NO_FACE, 0, None)

    xs = np.array([lm.x for lm in faces[0]]) * frame_w
    ys = np.array([lm.y for lm in faces[0]]) * frame_h
    bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    status = FaceStatus.OK if len(faces) == 1 else FaceStatus.MULTIPLE
    return FaceReading(status, len(faces), bbox)
