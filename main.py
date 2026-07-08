"""ProcVision — full application.

Run:  python main.py

Controls:
  Start/Stop button (or SPACE)  begin / pause the session
  Proctor / Class tabs (or TAB) switch dashboard mode
  R                             save session report (PNG + JSON) to reports/
  ESC                           quit (report auto-saves if a session ran)

The session clock, score history, and events only advance while the
session is running. All processing happens on-device; nothing is
recorded or uploaded.
"""

import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from proctor import ui
from proctor.blink import BlinkTracker, EyeState, read_blink
from proctor.events import EventDetector, FrameObservation
from proctor.face_detection import FaceStatus, read_face
from proctor.gaze import GazeDirection, GazeTracker
from proctor.head_pose import HeadDirection, read_head_pose
from proctor.landmarker import Landmarker, open_camera
from proctor.scoring import ConcentrationScorer

REPORTS_DIR = Path(__file__).resolve().parent / "reports"
WINDOW = "ProcVision"


class Session:
    """Pause-aware session clock and recorded data."""

    def __init__(self):
        self.running = False
        self._accumulated = 0.0
        self._run_started: float | None = None
        self.detector = EventDetector()
        self.scorer = ConcentrationScorer()
        self.gaze_tracker = GazeTracker()
        self.blink_tracker = BlinkTracker()
        self.times: list[float] = []
        self.scores: list[float] = []
        self.drowsy_spans: list[tuple[float, float | None]] = []
        self._drowsy_open = False

    @property
    def elapsed(self) -> float:
        if self.running and self._run_started is not None:
            return self._accumulated + (time.monotonic() - self._run_started)
        return self._accumulated

    def toggle(self) -> None:
        if self.running:
            self._accumulated = self.elapsed
            self._run_started = None
            self.running = False
        else:
            self._run_started = time.monotonic()
            self.running = True

    def note_drowsy(self, is_drowsy: bool, t: float) -> None:
        if is_drowsy and not self._drowsy_open:
            self.drowsy_spans.append((t, None))
            self._drowsy_open = True
        elif not is_drowsy and self._drowsy_open:
            s, _ = self.drowsy_spans[-1]
            self.drowsy_spans[-1] = (s, t)
            self._drowsy_open = False


def save_report(session: Session) -> Path:
    REPORTS_DIR.mkdir(exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    avg = float(np.mean(session.scores)) if session.scores else 0.0

    img = ui.render_report(
        generated_at=generated,
        duration_s=session.elapsed,
        avg_score=avg,
        events=session.detector.events,
        times=session.times,
        scores=session.scores,
        drowsy_spans=session.drowsy_spans,
    )
    png_path = REPORTS_DIR / f"session_{stamp}.png"
    cv2.imwrite(str(png_path), img)

    report = {
        "generated_at": generated,
        "session_duration_s": round(session.elapsed, 1),
        "average_concentration": round(avg, 1),
        "event_counts": session.detector.summary(),
        "events": [
            {
                "kind": e.kind,
                "detail": e.detail,
                "severity": e.severity.value,
                "start_s": round(e.t_start, 1),
                "end_s": round(e.t_end, 1) if e.t_end else None,
                "duration_s": round(e.duration, 1) if e.t_end else None,
            }
            for e in session.detector.events
        ],
    }
    (REPORTS_DIR / f"session_{stamp}.json").write_text(json.dumps(report, indent=2))
    return png_path


def main() -> None:
    landmarker = Landmarker(max_faces=2)
    cap = open_camera()
    session = Session()

    state = {"tab": "proctor", "clicked": None}

    def on_mouse(event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if ui.hit(ui.BTN_RECT, x, y):
            state["clicked"] = "toggle"
        elif ui.hit(ui.TAB_PROCTOR_RECT, x, y):
            state["tab"] = "proctor"
        elif ui.hit(ui.TAB_CLASS_RECT, x, y):
            state["tab"] = "class"

    cv2.namedWindow(WINDOW, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WINDOW, on_mouse)

    footer_msg = ""
    footer_msg_until = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera frame not available, stopping.")
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]
        t = session.elapsed

        if state["clicked"] == "toggle":
            session.toggle()
            state["clicked"] = None

        face = None
        gaze = head = None
        eye_state = None
        calibrating = False

        if session.running:
            result = landmarker.detect(frame)
            face = read_face(result, w, h)

            if face.status != FaceStatus.NO_FACE:
                lms = result.face_landmarks[0]
                gaze = session.gaze_tracker.read(lms, w, h, now=t)
                raw_blink = read_blink(lms, w, h)
                eye_state = session.blink_tracker.update(raw_blink, t)
                if result.facial_transformation_matrixes:
                    head = read_head_pose(result.facial_transformation_matrixes[0])

            calibrating = bool(gaze and gaze.calibrating)

            # Head and eyes counter-rotate: opposite directions mean the
            # eyes are still on the screen (see scoring.py).
            compensating = bool(gaze and head) and (
                (head.direction == HeadDirection.LEFT and gaze.direction == GazeDirection.RIGHT)
                or (head.direction == HeadDirection.RIGHT and gaze.direction == GazeDirection.LEFT)
                or (head.direction == HeadDirection.UP and gaze.direction == GazeDirection.DOWN)
                or (head.direction == HeadDirection.DOWN and gaze.direction == GazeDirection.UP)
            )
            # EAR is unreliable at high yaw and during deep look-down
            # (narrowed lids read as closed) — don't trust closure there.
            eyes_reliable = not (
                (head and head.direction in (HeadDirection.LEFT, HeadDirection.RIGHT))
                or (gaze and gaze.direction == GazeDirection.DOWN)
            )

            obs = FrameObservation(
                face_absent=(face.status == FaceStatus.NO_FACE),
                multiple_faces=(face.status == FaceStatus.MULTIPLE),
                gaze_off=(not calibrating and gaze is not None
                          and gaze.direction != GazeDirection.CENTER and not compensating),
                gaze_detail=gaze.direction.value if gaze else "",
                head_turned=(head is not None
                             and head.direction in (HeadDirection.LEFT, HeadDirection.RIGHT)
                             and not compensating),
                head_detail=head.direction.value if head else "",
                eyes_closed=(eyes_reliable and eye_state == EyeState.CLOSED),
                drowsy=(eyes_reliable and eye_state == EyeState.DROWSY),
            )
            session.detector.update(obs, t)

            score = session.scorer.update(
                gaze.direction if gaze else None,
                eye_state,
                head.direction if head else None,
                face_present=(face.status != FaceStatus.NO_FACE),
            )
            session.times.append(t)
            session.scores.append(score)
            session.note_drowsy(eye_state in (EyeState.DROWSY, EyeState.CLOSED), t)

            # video overlays
            if face.bbox:
                x1, y1, x2, y2 = face.bbox
                box_color = (0, 200, 90) if not session.detector.active_flags else (80, 80, 240)
                cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            if gaze:
                for iris in (gaze.right_iris_px, gaze.left_iris_px):
                    cv2.circle(frame, iris, 3, (200, 200, 60), -1)
            if calibrating:
                cv2.putText(frame, "calibrating gaze... look at the screen", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (60, 200, 255), 2, cv2.LINE_AA)
            elif face.status == FaceStatus.NO_FACE:
                cv2.putText(frame, "No face detected", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 240), 2, cv2.LINE_AA)
            elif face.status == FaceStatus.MULTIPLE:
                cv2.putText(frame, f"{face.count} faces in frame", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (80, 80, 240), 2, cv2.LINE_AA)
        else:
            label = "Press Start to begin the session" if not session.times else "Session paused"
            cv2.putText(frame, label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (60, 200, 255), 2, cv2.LINE_AA)

        # --- compose dashboard ---
        canvas = ui.make_canvas()
        ui.draw_header(canvas, state["tab"], t, session.running)
        ui.draw_video(canvas, frame)

        score_now = session.scores[-1] if session.scores else 0.0
        gaze_value = "calibrating..." if calibrating else (gaze.direction.value if gaze else "-")
        chips = [
            ("Face", face.status.value if face else "-", bool(face and face.status == FaceStatus.OK)),
            ("Gaze", gaze_value, bool(gaze and not calibrating and gaze.direction == GazeDirection.CENTER)),
            ("Head", head.direction.value if head else "-", bool(head and head.direction == HeadDirection.FORWARD)),
            ("Eyes", eye_state.value if eye_state else "-", eye_state == EyeState.OPEN),
        ]

        y = ui.PANEL_Y
        if state["tab"] == "class":
            y = ui.draw_gauge_card(canvas, y, score_now)
            y = ui.draw_chips(canvas, y, chips)
            remaining = ui.PANEL_Y + ui.VIDEO_H - y
            ui.draw_sparkline_card(canvas, y, remaining, session.times, session.scores,
                                   session.drowsy_spans, t)
        else:
            y = ui.draw_integrity_card(canvas, y, session.detector.active_flags,
                                       len(session.detector.events))
            y = ui.draw_chips(canvas, y, chips)
            remaining = ui.PANEL_Y + ui.VIDEO_H - y
            y = ui.draw_event_log_card(canvas, y, remaining - 76,
                                       session.detector.events, 0.0)
            ui.draw_timeline_card(canvas, y, session.detector.events, 0.0, max(t, 1.0))

        if time.monotonic() >= footer_msg_until:
            footer_msg = ""
        ui.draw_footer(canvas, footer_msg)

        cv2.imshow(WINDOW, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == 32:  # SPACE
            session.toggle()
        elif key == 9:  # TAB
            state["tab"] = "class" if state["tab"] == "proctor" else "proctor"
        elif key in (ord("r"), ord("R")):
            path = save_report(session)
            footer_msg = f"saved {path.name}"
            footer_msg_until = time.monotonic() + 3.0
            print(f"Report saved: {path}")

        if cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    if session.times:
        path = save_report(session)
        print(f"Session report saved: {path}")

    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
