"""Puts the two halves of the camera pipeline together: fingertips (MediaPipe) projected through
the neck homography (ArUco) become string + fret positions."""

from dataclasses import dataclass

import numpy as np

from .fretboard import FretPosition, locate, string_spacing
from .hands import FINGER_NUMBERS, FINGER_TIPS, Hand
from .instruments import Instrument
from .neck import NeckPose


@dataclass(frozen=True)
class FingerPlacement:
    finger: str  # "index" | "middle" | "ring" | "pinky"
    number: int  # 1–4, how teachers number fretting fingers
    x_mm: float
    y_mm: float
    pos: FretPosition
    on_string: bool  # on the neck and close to a string's centre (not between two strings)

    def describe(self, inst: Instrument) -> str:
        return f"finger {self.number} on {inst.string_label(self.pos.string)}, fret {self.pos.fret}"


@dataclass
class HandReading:
    fingers: dict[str, FingerPlacement]
    hand: Hand
    hand_size_px: float

    @property
    def on_neck(self) -> list[FingerPlacement]:
        return [f for f in self.fingers.values() if f.on_string]


def read_hand(hand: Hand, pose: NeckPose, inst: Instrument) -> HandReading:
    names = list(FINGER_TIPS)
    tips_mm = pose.to_neck(np.array([hand.tip(n) for n in names]))
    fingers = {}
    for name, (x, y) in zip(names, tips_mm):
        pos = locate(inst, float(x), float(y))
        on_string = pos.on_neck and abs(pos.string_offset_mm) <= 0.45 * string_spacing(inst, float(x))
        fingers[name] = FingerPlacement(name, FINGER_NUMBERS[name], float(x), float(y), pos, on_string)
    lm = hand.landmarks_px
    size = float(np.linalg.norm(lm[0, :2] - lm[9, :2]))  # wrist → middle knuckle
    return HandReading(fingers, hand, size)


def fretting_hand(hands: list[Hand], pose: NeckPose, inst: Instrument) -> HandReading | None:
    """The fretting hand is the one with the most fingertips on the fretboard
    (more robust than MediaPipe's left/right label, which flips with camera mirroring)."""
    best, best_score = None, 0
    for h in hands:
        reading = read_hand(h, pose, inst)
        score = sum(f.pos.on_neck for f in reading.fingers.values())
        if score > best_score:
            best, best_score = reading, score
    return best
