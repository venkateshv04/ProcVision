# ProcVision

You're on a video call, or halfway through a timed exam, and a notification pulls your eyes away for three seconds. It happens forty times an hour and nobody, not even you, has an honest sense of how present you actually were. ProcVision watches for exactly that, using nothing but a webcam and a face-landmark model: no keyboard logging, no browser snooping, just attention, the way a human proctor in the room would notice it, minus the fatigue of watching for hours straight.

**[Try it live](https://procvision.vercel.app)**, it runs entirely in your browser, nothing is recorded or uploaded. Sit centered in the frame, in decent light, and it will start reading your attention within a couple of seconds.

This repository is the Python reference implementation the live demo is built from.

## What it actually does

Point a webcam at yourself and the system tells you, in real time, whether you're paying attention to the screen, and if not, why: looking away, head turned, drowsy, asleep, absent, or someone else has entered the frame.

Two modes sit on top of the same engine:

- **Proctor mode**, built for remote exams. A sustained pattern (gaze held off-screen, head turned away, a second face appearing) becomes a timestamped event on a session timeline, and the whole session exports as a report.
- **Class mode**, built for lectures and self-study. A live 0 to 100 concentration score with a scrolling graph, so you can see your own focus trail off over an hour instead of just feeling tired afterward.

## How it works

```
Webcam frame
   -> MediaPipe Face Landmarker (478 landmarks + 3D transformation matrix)
        -> Face detected?  -- no --> "face absent / multiple faces" event
             |  yes
             +- Gaze       iris position relative to the eye corners -> direction
             +- Head pose  yaw / pitch / roll from the transformation matrix
             +- Blink      Eye Aspect Ratio -> open / drowsy / closed
                  |
             Event layer   a condition must PERSIST before it becomes an event
             Scoring       weighted fusion -> concentration index (0 to 100)
                  |
             Dashboard     live chips, event log, timeline, graph, report export
```

One model produces every signal:

| Signal | How it is measured |
|---|---|
| Face presence | Landmark detection succeeds or fails; a second face is itself a flag |
| Gaze direction | Iris center relative to the eye corners (horizontal) and the corner line (vertical) |
| Head pose | Yaw, pitch, and roll pulled straight from the landmarker's 3D transformation matrix, no manual camera calibration needed |
| Drowsiness | Eye Aspect Ratio: the eye opening narrows toward zero as the eyes close |

## The interesting part: edge cases

A naive version of this system, scoring each frame in isolation, sounds simple but breaks constantly. Getting it to behave sensibly is most of the actual engineering, so here is what breaks and why the fix is correct, not just convenient.

**A blink looks exactly like glancing down.** When your eyes close, the eyelids narrow and the iris marker drifts downward for a few frames, which is geometrically indistinguishable from checking a phone in your lap. The fix is a time gate: a downward gaze only counts once it holds for 0.3 seconds. A blink transition is over in under 0.15 seconds, so the two never collide.

**A blink also looks like drowsiness.** Everyone blinks every three to five seconds, and each one dips the eye-openness reading into the "closed" range for an instant. If that were treated as sleepiness, the score would flicker to zero constantly. Closure only counts after 0.4 seconds of holding, and a persistently low but open reading counts as drowsy only after 0.8 seconds, both comfortably past the length of an ordinary blink.

**Turning your head is not the same as looking away.** When you turn your head, your eyes counter-rotate to keep looking at the same spot, a reflex everyone has. So gaze measured relative to the face, not the screen, will read "looking right" the instant your head turns left, even though your eyes never left the screen. The scorer checks for this: opposite head and gaze directions cancel out and the system treats it as attention, exactly as a human observer would.

**Some readings should not be trusted at all.** At a sharp head angle, the eye landmarks foreshorten and the openness measurement collapses, which looks identical to falling asleep. A deep glance down does the same thing to the eyelids. Rather than let a broken measurement corrupt the score, eye-closure readings are simply ignored whenever the head is turned or the gaze is down, the same way you would not trust a fogged-up sensor.

**No two people's eyes sit the same way.** Where your iris rests when you're looking dead center depends on your anatomy and where the camera sits, so a fixed threshold works for some people and misfires constantly for others. Every session opens with about 1.5 seconds of calibration that learns your own neutral position, and every later reading is judged against that baseline instead of a population average.

**A number sitting right on a threshold flickers.** Gaze hovering at the boundary between "center" and "right" will cross back and forth every frame from ordinary sensor noise. Readings are smoothed first, and the direction only changes once it clears the threshold with room to spare, so it takes real movement to flip the label, not noise.

**A glance is not an incident.** Nobody stares at a screen without ever looking away, so nothing is logged until a condition has actually persisted: gaze off-screen for 1.5 seconds, no face for 1 second, eyes closed for 2 seconds. Each logged event also records its full duration, so a report reads as a timeline of real behavior, not a wall of noise.

## Getting started

Needs Python 3.10 or newer and a webcam. The landmark model (about 4 MB) downloads itself the first time you run it.

```bash
git clone https://github.com/venkateshv04/ProcVision.git
cd ProcVision
pip install -r requirements.txt
python main.py
```

| Control | Action |
|---|---|
| Start / Stop button (or `SPACE`) | Begin or pause the session |
| Tabs (or `TAB`) | Switch between Proctor mode and Class mode |
| `R` | Save the session report (PNG and JSON) to `reports/` |
| `ESC` | Quit (the report saves automatically) |

Sit roughly centered in the frame with your face fully visible and well lit. Calibration and head pose both depend on a frontal, evenly lit view, on the live demo too.

A few things worth trying once it is running: glance down at your phone under the desk and watch a "Gaze off-screen, looking down" event appear after about 1.5 seconds; turn your head while keeping your eyes on the screen and notice nothing gets flagged; close your eyes for three seconds and watch the score drop to zero; have someone step into frame and watch a critical "Multiple faces" event fire immediately.

### Each signal on its own

Every detector is a standalone module with its own visual demo, useful for understanding one signal in isolation or for debugging it.

```bash
python demos/demo_face.py        # bounding box, landmark mesh, face count
python demos/demo_gaze.py        # iris crosshairs, direction, live calibration values
python demos/demo_head_pose.py   # 3D axes on the nose, yaw/pitch/roll readout
python demos/demo_blink.py       # eye contours, per-eye EAR, blink state
```

These demos import the same modules the main app runs. If a demo works, the app works.

## Project structure

```
├── proctor/               core package, logic kept separate from UI
│   ├── landmarker.py      MediaPipe wrapper and model auto-download
│   ├── face_detection.py  presence, count, bounding box
│   ├── gaze.py            iris gaze with per-user auto-calibration
│   ├── head_pose.py       yaw, pitch, roll
│   ├── blink.py           EAR-based blink and drowsiness, time-debounced
│   ├── events.py          the sustained-condition event state machine
│   ├── scoring.py         weighted concentration index
│   ├── config.py          every tunable threshold, in one place
│   └── ui.py              dashboard rendering
├── demos/                 one standalone camera demo per signal
├── main.py                the full two-mode application
└── requirements.txt
```

## Scoring

| Signal | Weight | Full credit | Half credit | Zero |
|---|---|---|---|---|
| Gaze | 0.4 | eyes on screen (centered, or head-compensated) | none | looking away |
| Drowsiness | 0.3 | eyes open | drowsy | closed |
| Head orientation | 0.3 | facing forward | tilted up or down | turned away with eyes off screen |

Sustained, reliably-measured eye closure overrides everything and drops the score to zero. What's displayed is smoothed with an exponential moving average, so it moves like a gauge settling, not a needle jumping around.

Every threshold behind these numbers, gaze deltas, EAR bounds, debounce timings, event durations, weights, lives in [`proctor/config.py`](proctor/config.py) with a comment explaining the reasoning behind each one.

## Where this falls short

Behavioral monitoring produces evidence for a human to review, not a verdict.

- Poor lighting, backlight, glasses glare, and anything covering part of the face all degrade landmark accuracy. Use even light facing the camera.
- The camera should be roughly frontal with the person centered; an off-angle or off-center view shifts what "neutral" head pose even means.
- Calibration assumes you're looking at the screen the moment a session starts.
- Looking down could be cheating, or it could be taking notes. The system reports what it sees; deciding what it means is a human's job.

## Tech stack

MediaPipe's Face Landmarker, OpenCV, and NumPy. The live demo is a TypeScript port of this exact pipeline, compiled against MediaPipe's WebAssembly runtime and deployed on Vercel, so it runs on-device in a browser with no server involved.
