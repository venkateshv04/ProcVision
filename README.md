# ProcVision — Visual Proctoring & Focus Analysis

Real-time behavioral analysis from a webcam, built entirely with computer vision. One MediaPipe Face Landmarker pass drives four signals — face presence, gaze direction, head pose, and blink/drowsiness — which feed a temporal event detector and a weighted concentration index.

Two modes, switchable live:

- **Proctor mode** — integrity signals for remote assessments: sustained gaze-off-screen, head turned away, face absent, multiple faces, prolonged eye closure. Events are logged with timestamps and drawn on a session timeline.
- **Class mode** — the original concentration-index concept: a weighted 0–100 focus score with drowsiness tracking and a live scrolling graph.

> This is the **Python reference implementation**. A browser port (same pipeline, running fully on-device via WebAssembly) lives in `../web_version` and is deployed as a live demo.

## How it works

```
Webcam frame
   └─ MediaPipe Face Landmarker (478 landmarks + transformation matrix)
        └─ Face detected?  ── no ──► "face absent / multiple faces" flag
             │ yes
             ├─ gaze.py        iris position vs eye corners/lids → direction
             ├─ head_pose.py   yaw/pitch/roll from transformation matrix
             └─ blink.py       Eye Aspect Ratio → open / drowsy / closed
                  │
             events.py         sustained-deviation state machine (hysteresis)
             scoring.py        weighted fusion → concentration index (EMA-smoothed)
                  │
             main.py           live dashboard: tabs, event log, timeline, graph
```

Design points worth noting:

- **Single model, four signals.** No dlib, no separate gaze library, no 68 MB landmark file — the ~4 MB `.task` model is auto-downloaded on first run.
- **Temporal hysteresis, not per-frame flags.** A glance to the side is not an event; gaze held off-screen for 1.5 s is. This kills the false positives that plague naive per-frame scoring (a slow blink is not "sleeping").
- **Head pose without solvePnP.** Orientation comes from the landmarker's facial transformation matrix — no hand-built camera intrinsics to get wrong.
- **Every threshold lives in `proctor/config.py`** — no magic numbers scattered through the code.
- **Privacy by architecture.** Frames are processed in memory and discarded. Nothing is recorded or transmitted; the only output is an optional JSON event report you save yourself.

## Install

Requires Python 3.10+ and a webcam.

```bash
git clone <repo-url>
cd procvision/python_version
pip install -r requirements.txt
```

## Run

Full application (both modes):

```bash
python main.py
```

| Key | Action |
|-----|--------|
| `TAB` | Switch between Proctor mode and Class mode |
| `R` | Save session report (JSON) to `reports/` |
| `ESC` | Quit (report auto-saves on exit) |

### Individual module demos

Each signal is independently runnable — the camera opens and only that module's output is visualized:

```bash
python demos/demo_face.py        # bounding box, landmark mesh, face count
python demos/demo_gaze.py        # pupil crosshairs, direction, raw ratios
python demos/demo_head_pose.py   # 3D axes on the nose, yaw/pitch/roll
python demos/demo_blink.py       # eye contours, per-eye EAR, drowsy state
```

The demos import the exact same modules the main app uses — if a demo works, the app works.

## Project structure

```
python_version/
├── proctor/               core package (pure logic, UI separated)
│   ├── landmarker.py      MediaPipe wrapper + model auto-download
│   ├── face_detection.py  presence / count / bounding box
│   ├── gaze.py            iris-based gaze direction
│   ├── head_pose.py       yaw / pitch / roll
│   ├── blink.py           EAR-based blink & drowsiness
│   ├── events.py          sustained-deviation event state machine
│   ├── scoring.py         weighted concentration index
│   ├── config.py          all tunable thresholds
│   └── ui.py              dashboard rendering (OpenCV)
├── demos/                 one standalone camera demo per signal
├── main.py                full two-mode application
└── requirements.txt
```

## Concentration index

Same weighted-fusion concept as the original project:

| Signal | Weight | Full credit | Half credit | Zero |
|--------|--------|-------------|-------------|------|
| Gaze | 0.4 | looking center | — | any other direction |
| Drowsiness | 0.3 | eyes open | drowsy | closed |
| Head orientation | 0.3 | facing forward | up / down | left / right |

Eyes fully closed overrides the score to 0. The displayed value is smoothed with an exponential moving average.

## Limitations (read before judging a student with this)

Behavioral proctoring produces **evidence for a human reviewer, not verdicts**. Known failure modes:

- Poor lighting, strong backlight, glasses glare, and occlusions degrade landmark accuracy.
- Vertical gaze auto-calibrates against the user's neutral eye position during the first ~1.5 s of a session — the user should be looking at the screen when it starts.
- Looking down can mean cheating — or taking notes. Context matters; the tool reports, humans decide.
- Camera angle changes the neutral head pose; users should frame themselves roughly frontal.

## Roadmap

- [x] Per-user vertical gaze auto-calibration at session start
- [ ] Full "follow the dot" calibration for horizontal thresholds too
- [ ] Web version (browser port, fully on-device) — `../web_version`
- [ ] Configurable sensitivity presets (demo / lenient / strict)

## Lineage

Evolved from a 2024 mini-project ("Concentration Index in Online Learning Environments", dlib + separate gaze library + Haar cascades) — rebuilt on a single-model MediaPipe pipeline with a temporal event layer and proper module separation.
