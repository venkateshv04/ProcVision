"""Standalone demo: head pose only.

Run:  python demos/demo_head_pose.py
Shows 3D axes projected from the nose, yaw/pitch/roll values, and the
facing direction. ESC to quit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from proctor.head_pose import HeadDirection, axis_endpoints, read_head_pose
from proctor.landmarker import Landmarker, open_camera, to_px

NOSE_TIP = 1


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
        if result.face_landmarks and result.facial_transformation_matrixes:
            reading = read_head_pose(result.facial_transformation_matrixes[0])
            nose = to_px(result.face_landmarks[0][NOSE_TIP], w, h)

            axes = axis_endpoints(reading, nose)
            cv2.line(frame, nose, axes["x"], (0, 0, 255), 2, cv2.LINE_AA)   # x red
            cv2.line(frame, nose, axes["y"], (0, 255, 0), 2, cv2.LINE_AA)   # y green
            cv2.line(frame, nose, axes["z"], (255, 0, 0), 2, cv2.LINE_AA)   # z blue

            color = (0, 255, 0) if reading.direction == HeadDirection.FORWARD else (0, 0, 255)
            cv2.putText(frame, reading.direction.value, (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
            cv2.putText(frame,
                        f"yaw: {reading.yaw:+.1f}  pitch: {reading.pitch:+.1f}  roll: {reading.roll:+.1f}",
                        (10, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, "No face detected", (10, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        cv2.imshow("Demo - Head pose", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
