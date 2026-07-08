"""Standalone demo: blink / drowsiness only.

Run:  python demos/demo_blink.py
Shows eye contours, per-eye EAR values, and the open/drowsy/closed
state. ESC to quit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np

from proctor.blink import EyeState, read_blink
from proctor.landmarker import Landmarker, open_camera

STATE_COLORS = {
    EyeState.OPEN: (0, 255, 0),
    EyeState.DROWSY: (0, 255, 255),
    EyeState.CLOSED: (0, 0, 255),
}


def main() -> None:
    landmarker = Landmarker(max_faces=1)
    cap = open_camera()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame not available, stopping.")
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        result = landmarker.detect(frame)
        if result.face_landmarks:
            reading = read_blink(result.face_landmarks[0], w, h)
            color = STATE_COLORS[reading.state]

            for eye in (reading.right_eye_px, reading.left_eye_px):
                cv2.polylines(frame, [np.array(eye, dtype=np.int32)], True, color, 1, cv2.LINE_AA)

            cv2.putText(frame, reading.state.value, (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            cv2.putText(frame,
                        f"EAR right: {reading.ear_right:.3f}  left: {reading.ear_left:.3f}  avg: {reading.ear_avg:.3f}",
                        (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No face detected", (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Demo - Blink / drowsiness", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
