"""Draws what guitra sees on top of the camera image: markers, the fretboard grid,
each fingertip's string/fret, recent notes and alerts."""

import cv2
import numpy as np

from .fretboard import fret_x, string_y
from .instruments import Instrument
from .mapping import HandReading
from .neck import NeckPose
from .notes import midi_to_name

AMBER = (40, 170, 245)
TEAL = (190, 200, 60)
WHITE = (240, 240, 240)
RED = (60, 60, 230)
GREY = (150, 150, 150)


def draw(
    frame: np.ndarray,
    inst: Instrument,
    pose: NeckPose | None,
    detections: list[tuple[int, np.ndarray]],
    hand: HandReading | None,
    ringing: list[int],
    alerts: list[str],
    status: str,
) -> np.ndarray:
    out = frame.copy()
    for marker_id, px in detections:
        cv2.polylines(out, [px.astype(np.int32)], True, TEAL, 2)
        cv2.putText(out, str(marker_id), tuple(px[0].astype(int)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEAL, 2)

    if pose is not None:
        _draw_grid(out, inst, pose)

    if hand is not None:
        for f in hand.fingers.values():
            tip = hand.hand.tip(f.finger).astype(int)
            color = AMBER if f.on_string else GREY
            cv2.circle(out, tuple(tip), 7, color, -1)
            label = f"{f.number}: s{inst.strings - f.pos.string} f{f.pos.fret}" if f.on_string else f"{f.number}: off"
            cv2.putText(out, label, (tip[0] + 9, tip[1] - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    h, w = out.shape[:2]
    cv2.rectangle(out, (0, 0), (w, 28), (20, 20, 20), -1)
    heard = " ".join(midi_to_name(m) for m in sorted(ringing)) or "-"
    cv2.putText(out, f"{status}   heard: {heard}", (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1)
    lines = [line for msg in alerts[-3:] for line in _wrap(msg, max(40, w // 10))]
    for i, line in enumerate(lines):
        y = h - 12 - 24 * (len(lines) - 1 - i)
        cv2.rectangle(out, (0, y - 18), (w, y + 7), (25, 25, 60), -1)
        cv2.putText(out, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 210, 255), 1)
    return out


def _draw_grid(img: np.ndarray, inst: Instrument, pose: NeckPose) -> None:
    color = TEAL if pose.reliable and pose.age_frames == 0 else GREY
    last = inst.strings - 1
    for n in range(inst.frets + 1):
        x = fret_x(inst, n)
        a, b = pose.to_image(np.array([[x, string_y(inst, 0, x)], [x, string_y(inst, last, x)]]))
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, 2 if n == 0 else 1)
    end = fret_x(inst, inst.frets)
    for s in range(inst.strings):
        a, b = pose.to_image(np.array([[0.0, string_y(inst, s, 0.0)], [end, string_y(inst, s, end)]]))
        cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), color, 1)


def _wrap(text: str, width: int) -> list[str]:
    lines, cur = [], ""
    for word in text.split():
        if cur and len(cur) + 1 + len(word) > width:
            lines.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    return lines + ([cur] if cur else [])
