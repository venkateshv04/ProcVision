"""Standalone demo: gaze direction only.

Run:  python demos/demo_gaze.py
Shows pupil crosshairs, gaze direction label, and raw ratios. ESC to quit.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from proctor.gaze import GazeDirection, GazeTracker
from proctor.landmarker import Landmarker, open_camera


def crosshair(frame, center, color, size=8):
    x, y = center
    cv2.line(frame, (x - size, y), (x + size, y), color, 1, cv2.LINE_AA)
    cv2.line(frame, (x, y - size), (x, y + size), color, 1, cv2.LINE_AA)


def main() -> None:
    landmarker = Landmarker(max_faces=1)
    tracker = GazeTracker()
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
            reading = tracker.read(result.face_landmarks[0], w, h, now=time.monotonic())
            if reading.calibrating:
                color = (0, 255, 255)
                label = "calibrating... look at the screen"
            else:
                color = (0, 255, 0) if reading.direction == GazeDirection.CENTER else (0, 0, 255)
                label = reading.direction.value

            crosshair(frame, reading.right_iris_px, color)
            crosshair(frame, reading.left_iris_px, color)

            cv2.putText(frame, label, (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            baseline = f"{reading.v_baseline:+.3f}" if reading.v_baseline is not None else "..."
            o_base = tracker.openness_baseline
            o_ratio = f"{reading.openness / o_base:.2f}" if o_base else "..."
            cv2.putText(frame,
                        f"h: {reading.h_ratio:.2f}  v: {reading.v_ratio:+.3f}  base: {baseline}",
                        (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            cv2.putText(frame,
                        f"openness: {reading.openness:.3f}  ratio vs base: {o_ratio}",
                        (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No face detected", (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Demo - Gaze direction", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
