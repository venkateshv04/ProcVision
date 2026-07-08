"""Dashboard rendering (OpenCV canvas).

Layout contract — the web UI replicates this exactly:

  +--------------------------------------------------------------+
  | header: app name          [ Proctor mode | Class mode ]  MM:SS |
  +---------------------------------------+----------------------+
  |                                       |  side panel (400 px) |
  |   video 640x480 with overlays         |  cards per tab       |
  |                                       |                      |
  +---------------------------------------+----------------------+
  | footer: key hints                                             |
  +--------------------------------------------------------------+

  Proctor tab: integrity card, 2x2 status chips, event log, timeline
  Class tab:   concentration gauge, 2x2 status chips, live sparkline
"""

from __future__ import annotations

import cv2
import numpy as np

from .events import Event, Severity

# --- layout ---
MARGIN = 12
VIDEO_W, VIDEO_H = 640, 480
PANEL_W = 400
HEADER_H = 52
FOOTER_H = 34
CANVAS_W = MARGIN + VIDEO_W + MARGIN + PANEL_W + MARGIN
CANVAS_H = HEADER_H + MARGIN + VIDEO_H + MARGIN + FOOTER_H
VIDEO_X, VIDEO_Y = MARGIN, HEADER_H + MARGIN
PANEL_X = MARGIN + VIDEO_W + MARGIN
PANEL_Y = HEADER_H + MARGIN

# --- palette (BGR) ---
C_BG = (26, 22, 20)
C_HEADER = (38, 32, 30)
C_CARD = (48, 41, 38)
C_CARD_BORDER = (68, 60, 56)
C_TEXT = (240, 240, 240)
C_MUTED = (155, 150, 145)
C_OK = (128, 222, 74)
C_WARN = (36, 191, 251)
C_BAD = (113, 113, 248)
C_ACCENT = (166, 184, 20)

FONT = cv2.FONT_HERSHEY_SIMPLEX

# --- header hit areas (x, y, w, h) for mouse clicks ---
_TAB_W, _TAB_H = 150, 34
_CLOCK_W = 80
_TABS_X = CANVAS_W - MARGIN - _CLOCK_W - 2 * _TAB_W - 6 - 10
_HDR_Y = (HEADER_H - _TAB_H) // 2
BTN_RECT = (_TABS_X - 110 - 16, _HDR_Y, 110, _TAB_H)
TAB_PROCTOR_RECT = (_TABS_X, _HDR_Y, _TAB_W, _TAB_H)
TAB_CLASS_RECT = (_TABS_X + _TAB_W + 6, _HDR_Y, _TAB_W, _TAB_H)


def hit(rect: tuple[int, int, int, int], x: int, y: int) -> bool:
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def fmt_clock(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


def _text(img, s, org, scale=0.45, color=C_TEXT, thick=1):
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def make_canvas() -> np.ndarray:
    canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
    canvas[:] = C_BG
    return canvas


def draw_header(canvas: np.ndarray, active_tab: str, elapsed: float, running: bool) -> None:
    cv2.rectangle(canvas, (0, 0), (CANVAS_W, HEADER_H), C_HEADER, -1)
    _text(canvas, "ProcVision", (MARGIN + 4, 33), 0.75, C_TEXT, 2)
    _text(canvas, "on-device visual proctoring", (150, 33), 0.4, C_MUTED)

    # start / stop button
    bx, by, bw, bh = BTN_RECT
    btn_color = C_BAD if running else C_OK
    btn_label = "Stop" if running else "Start"
    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), btn_color, -1)
    size = cv2.getTextSize(btn_label, FONT, 0.55, 2)[0]
    _text(canvas, btn_label, (bx + (bw - size[0]) // 2, by + 23), 0.55, (20, 20, 20), 2)

    # tabs
    for rect, key, label in (
        (TAB_PROCTOR_RECT, "proctor", "Proctor mode"),
        (TAB_CLASS_RECT, "class", "Class mode"),
    ):
        x, y, tab_w, tab_h = rect
        active = key == active_tab
        fill = C_ACCENT if active else C_CARD
        text_color = (20, 20, 20) if active else C_MUTED
        cv2.rectangle(canvas, (x, y), (x + tab_w, y + tab_h), fill, -1)
        size = cv2.getTextSize(label, FONT, 0.5, 1)[0]
        _text(canvas, label, (x + (tab_w - size[0]) // 2, y + 23), 0.5, text_color, 1)

    _text(canvas, fmt_clock(elapsed), (CANVAS_W - MARGIN - 70, 34), 0.65, C_TEXT, 2)


def draw_video(canvas: np.ndarray, frame: np.ndarray) -> None:
    h, w = frame.shape[:2]
    if (w, h) != (VIDEO_W, VIDEO_H):
        frame = cv2.resize(frame, (VIDEO_W, VIDEO_H))
    canvas[VIDEO_Y:VIDEO_Y + VIDEO_H, VIDEO_X:VIDEO_X + VIDEO_W] = frame
    cv2.rectangle(canvas, (VIDEO_X - 1, VIDEO_Y - 1),
                  (VIDEO_X + VIDEO_W, VIDEO_Y + VIDEO_H), C_CARD_BORDER, 1)


def draw_footer(canvas: np.ndarray, message: str = "") -> None:
    y = CANVAS_H - FOOTER_H
    _text(canvas, "SPACE start/stop    TAB switch mode    R save report    ESC quit",
          (MARGIN + 4, y + 22), 0.42, C_MUTED)
    if message:
        size = cv2.getTextSize(message, FONT, 0.42, 1)[0]
        _text(canvas, message, (CANVAS_W - MARGIN - size[0], y + 22), 0.42, C_OK)


def _card(canvas: np.ndarray, y: int, h: int, title: str | None = None) -> tuple[int, int, int]:
    """Draw a card in the panel column. Returns (inner_x, inner_y, inner_w)."""
    x0, x1 = PANEL_X, PANEL_X + PANEL_W
    cv2.rectangle(canvas, (x0, y), (x1, y + h), C_CARD, -1)
    cv2.rectangle(canvas, (x0, y), (x1, y + h), C_CARD_BORDER, 1)
    inner_y = y + 8
    if title:
        _text(canvas, title.upper(), (x0 + 12, y + 20), 0.38, C_MUTED)
        inner_y = y + 30
    return x0 + 12, inner_y, PANEL_W - 24


def draw_chips(canvas: np.ndarray, y: int, chips: list[tuple[str, str, bool]]) -> int:
    """2x2 grid of status chips: (label, value, ok). Returns next free y."""
    chip_w = (PANEL_W - 8) // 2
    chip_h = 46
    for i, (label, value, ok) in enumerate(chips):
        cx = PANEL_X + (i % 2) * (chip_w + 8)
        cy = y + (i // 2) * (chip_h + 8)
        cv2.rectangle(canvas, (cx, cy), (cx + chip_w, cy + chip_h), C_CARD, -1)
        cv2.rectangle(canvas, (cx, cy), (cx + chip_w, cy + chip_h), C_CARD_BORDER, 1)
        dot = C_OK if ok else C_BAD
        if value.startswith("calibrating"):
            dot = C_WARN
        cv2.circle(canvas, (cx + 14, cy + 23), 5, dot, -1)
        _text(canvas, label.upper(), (cx + 26, cy + 18), 0.34, C_MUTED)
        _text(canvas, value[:22], (cx + 26, cy + 37), 0.45, C_TEXT)
    rows = (len(chips) + 1) // 2
    return y + rows * (chip_h + 8) + 4


def draw_integrity_card(canvas: np.ndarray, y: int, active_flags: list[str], total_events: int) -> int:
    h = 64
    ix, iy, iw = _card(canvas, y, h)
    flagged = len(active_flags) > 0
    color = C_BAD if flagged else C_OK
    label = "ATTENTION FLAGGED" if flagged else "ALL CLEAR"
    cv2.rectangle(canvas, (PANEL_X, y), (PANEL_X + 5, y + h), color, -1)
    _text(canvas, label, (ix + 4, y + 28), 0.62, color, 2)
    sub = ", ".join(active_flags) if flagged else f"{total_events} events this session"
    _text(canvas, sub[:46], (ix + 4, y + 50), 0.4, C_MUTED)
    return y + h + 10


def draw_gauge_card(canvas: np.ndarray, y: int, score: float) -> int:
    h = 86
    ix, iy, iw = _card(canvas, y, h, "concentration")
    color = C_OK if score >= 50 else C_BAD
    _text(canvas, f"{int(score)}%", (ix, y + 66), 1.15, color, 2)
    bar_x, bar_y = ix + 120, y + 48
    bar_w, bar_h = iw - 120, 16
    cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), C_BG, -1)
    fill = int(bar_w * min(max(score, 0.0), 100.0) / 100.0)
    if fill > 0:
        cv2.rectangle(canvas, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h), color, -1)
    mid = bar_x + bar_w // 2
    cv2.line(canvas, (mid, bar_y - 3), (mid, bar_y + bar_h + 3), C_MUTED, 1)
    return y + h + 10


def draw_sparkline_card(
    canvas: np.ndarray, y: int, h: int,
    times: list[float], values: list[float],
    drowsy_spans: list[tuple[float, float | None]],
    now: float, window_s: float = 60.0,
) -> int:
    ix, iy, iw = _card(canvas, y, h, "concentration - last 60 s")
    gx, gy = ix, iy + 4
    gw, gh = iw, h - (iy - y) - 26

    t_min = now - window_s

    def tx(t: float) -> int:
        return gx + int(max(0.0, min(1.0, (t - t_min) / window_s)) * gw)

    def vy(v: float) -> int:
        return gy + gh - int(min(max(v, 0.0), 100.0) / 100.0 * gh)

    # grid lines at 0 / 50 / 100
    for val in (0, 50, 100):
        ly = vy(val)
        cv2.line(canvas, (gx, ly), (gx + gw, ly), (60, 52, 48), 1)
        _text(canvas, str(val), (gx + gw - 24, ly - 3), 0.32, C_MUTED)

    # drowsy shading
    for s, e in drowsy_spans:
        e = e if e is not None else now
        if e < t_min:
            continue
        xs, xe = max(tx(s), gx), min(tx(e), gx + gw)
        if xe > xs:
            region = canvas[gy:gy + gh, xs:xe]
            red = np.zeros_like(region)
            red[:] = (40, 40, 110)
            cv2.addWeighted(red, 0.5, region, 0.5, 0, region)

    pts = [(tx(t), vy(v)) for t, v in zip(times, values) if t >= t_min]
    if len(pts) > 1:
        cv2.polylines(canvas, [np.array(pts, dtype=np.int32)], False, C_OK, 2, cv2.LINE_AA)

    return y + h + 10


def draw_event_log_card(canvas: np.ndarray, y: int, h: int, events: list[Event], session_start: float) -> int:
    ix, iy, iw = _card(canvas, y, h, "event log")
    row_h = 24
    max_rows = max((y + h - iy - 6) // row_h, 1)
    recent = events[-max_rows:]
    ry = iy + 14
    if not recent:
        _text(canvas, "no events flagged", (ix, ry + 4), 0.42, C_MUTED)
        return y + h + 10
    for e in recent:
        sev = {Severity.CRITICAL: C_BAD, Severity.WARNING: C_WARN, Severity.INFO: C_MUTED}[e.severity]
        cv2.circle(canvas, (ix + 5, ry - 4), 4, sev, -1)
        stamp = fmt_clock(e.t_start - session_start)
        tail = "ongoing" if e.t_end is None else f"{e.duration:.1f}s"
        _text(canvas, f"{stamp}  {e.kind}", (ix + 16, ry), 0.42, C_TEXT)
        size = cv2.getTextSize(tail, FONT, 0.38, 1)[0]
        _text(canvas, tail, (ix + iw - size[0], ry), 0.38, C_MUTED)
        ry += row_h
    return y + h + 10


def draw_timeline_card(canvas: np.ndarray, y: int, events: list[Event], session_start: float, now: float) -> int:
    h = 66
    ix, iy, iw = _card(canvas, y, h, "session timeline")
    strip_y, strip_h = iy + 4, 20
    cv2.rectangle(canvas, (ix, strip_y), (ix + iw, strip_y + strip_h), C_BG, -1)
    duration = max(now - session_start, 1.0)
    for e in events:
        s = (e.t_start - session_start) / duration
        t_end = e.t_end if e.t_end is not None else now
        t = (t_end - session_start) / duration
        xs = ix + int(s * iw)
        xe = max(ix + int(t * iw), xs + 2)
        color = C_BAD if e.severity == Severity.CRITICAL else C_WARN
        cv2.rectangle(canvas, (xs, strip_y + 2), (min(xe, ix + iw), strip_y + strip_h - 2), color, -1)
    _text(canvas, "00:00", (ix, strip_y + strip_h + 16), 0.35, C_MUTED)
    label = fmt_clock(now - session_start)
    size = cv2.getTextSize(label, FONT, 0.35, 1)[0]
    _text(canvas, label, (ix + iw - size[0], strip_y + strip_h + 16), 0.35, C_MUTED)
    return y + h + 10


# ---------------------------------------------------------------------------
# Session report image
# ---------------------------------------------------------------------------

def render_report(
    generated_at: str,
    duration_s: float,
    avg_score: float,
    events: list[Event],
    times: list[float],
    scores: list[float],
    drowsy_spans: list[tuple[float, float | None]],
) -> np.ndarray:
    """Render the whole session as a shareable report image."""
    W = 1000
    max_event_rows = 10
    n_rows = min(len(events), max_event_rows) or 1
    H = 520 + 30 + n_rows * 26 + 40
    img = np.zeros((H, W, 3), dtype=np.uint8)
    img[:] = C_BG

    def card(x, y, w, h, title=None):
        cv2.rectangle(img, (x, y), (x + w, y + h), C_CARD, -1)
        cv2.rectangle(img, (x, y), (x + w, y + h), C_CARD_BORDER, 1)
        if title:
            _text(img, title.upper(), (x + 12, y + 20), 0.38, C_MUTED)

    # header
    cv2.rectangle(img, (0, 0), (W, 64), C_HEADER, -1)
    _text(img, "ProcVision  -  session report", (20, 40), 0.8, C_TEXT, 2)
    size = cv2.getTextSize(generated_at, FONT, 0.45, 1)[0]
    _text(img, generated_at, (W - 20 - size[0], 40), 0.45, C_MUTED)

    # stat cards
    critical = sum(1 for e in events if e.severity == Severity.CRITICAL)
    stats = [
        ("duration", fmt_clock(duration_s)),
        ("avg concentration", f"{avg_score:.0f}%"),
        ("events flagged", str(len(events))),
        ("critical", str(critical)),
    ]
    sw = (W - 40 - 3 * 12) // 4
    for i, (label, value) in enumerate(stats):
        x = 20 + i * (sw + 12)
        card(x, 80, sw, 72, label)
        color = C_TEXT
        if label == "critical" and critical > 0:
            color = C_BAD
        if label == "avg concentration":
            color = C_OK if avg_score >= 50 else C_BAD
        _text(img, value, (x + 12, 80 + 56), 0.85, color, 2)

    # concentration graph, full session
    gy0 = 168
    card(20, gy0, W - 40, 230, "concentration over the session")
    gx, gy = 36, gy0 + 34
    gw, gh = W - 40 - 32, 230 - 34 - 26
    duration = max(duration_s, 1.0)

    def tx(t):
        return gx + int(min(max(t / duration, 0.0), 1.0) * gw)

    def vy(v):
        return gy + gh - int(min(max(v, 0.0), 100.0) / 100.0 * gh)

    for val in (0, 50, 100):
        ly = vy(val)
        cv2.line(img, (gx, ly), (gx + gw, ly), (60, 52, 48), 1)
        _text(img, str(val), (gx + gw + 4, ly + 4), 0.32, C_MUTED)
    for s, e in drowsy_spans:
        e = e if e is not None else duration
        xs, xe = tx(s), tx(e)
        if xe > xs:
            region = img[gy:gy + gh, xs:xe]
            red = np.zeros_like(region)
            red[:] = (40, 40, 110)
            cv2.addWeighted(red, 0.5, region, 0.5, 0, region)
    pts = [(tx(t), vy(v)) for t, v in zip(times, scores)]
    if len(pts) > 1:
        cv2.polylines(img, [np.array(pts, dtype=np.int32)], False, C_OK, 2, cv2.LINE_AA)

    # timeline
    ty0 = gy0 + 230 + 14
    card(20, ty0, W - 40, 78, "flagged intervals")
    strip_y = ty0 + 32
    cv2.rectangle(img, (36, strip_y), (W - 36, strip_y + 22), C_BG, -1)
    for e in events:
        xs = 36 + int((e.t_start / duration) * (W - 72))
        t_end = e.t_end if e.t_end is not None else duration
        xe = max(36 + int((t_end / duration) * (W - 72)), xs + 2)
        color = C_BAD if e.severity == Severity.CRITICAL else C_WARN
        cv2.rectangle(img, (xs, strip_y + 2), (min(xe, W - 36), strip_y + 20), color, -1)
    _text(img, "00:00", (36, strip_y + 38), 0.35, C_MUTED)
    end_label = fmt_clock(duration)
    size = cv2.getTextSize(end_label, FONT, 0.35, 1)[0]
    _text(img, end_label, (W - 36 - size[0], strip_y + 38), 0.35, C_MUTED)

    # event list
    ly0 = ty0 + 78 + 14
    list_h = 30 + n_rows * 26 + 12
    card(20, ly0, W - 40, list_h, "event log")
    ry = ly0 + 44
    if not events:
        _text(img, "no events flagged", (36, ry), 0.45, C_MUTED)
    for e in events[:max_event_rows]:
        sev = {Severity.CRITICAL: C_BAD, Severity.WARNING: C_WARN, Severity.INFO: C_MUTED}[e.severity]
        cv2.circle(img, (42, ry - 5), 4, sev, -1)
        dur = f"{e.duration:.1f}s" if e.t_end is not None else "ongoing"
        _text(img, f"{fmt_clock(e.t_start)}   {e.kind}  -  {e.detail}", (56, ry), 0.45, C_TEXT)
        size = cv2.getTextSize(dur, FONT, 0.42, 1)[0]
        _text(img, dur, (W - 36 - size[0], ry), 0.42, C_MUTED)
        ry += 26
    if len(events) > max_event_rows:
        _text(img, f"+ {len(events) - max_event_rows} more events (see JSON report)", (56, ry), 0.4, C_MUTED)

    return img
