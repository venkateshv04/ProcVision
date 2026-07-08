"""Standalone demo: face detection only.

Run:  python demos/demo_face.py
Shows bounding box, landmark dots, and face-count status. ESC to quit.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2

from proctor.face_detection import FaceStatus, read_face
from proctor.landmarker import Landmarker, open_camera, to_px


def main() -> None:
    landmarker = Landmarker(max_faces=2)
    cap = open_camera()

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame not available, stopping.")
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        result = landmarker.detect(frame)
        reading = read_face(result, w, h)

        if reading.status == FaceStatus.NO_FACE:
            banner, color = "No face detected", (0, 0, 255)
        elif reading.status == FaceStatus.MULTIPLE:
            banner, color = f"{reading.count} faces detected!", (0, 0, 255)
        else:
            banner, color = "1 face detected", (0, 255, 0)

        for face_lms in result.face_landmarks:
            for lm in face_lms[::8]:  # every 8th landmark, enough to see the mesh
                cv2.circle(frame, to_px(lm, w, h), 1, (200, 200, 60), -1)

        if reading.bbox:
            x1, y1, x2, y2 = reading.bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        cv2.putText(frame, banner, (10, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
        cv2.imshow("Demo - Face detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
