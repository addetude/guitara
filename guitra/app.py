"""The practice loop: camera frames + audio → neck pose, fretting hand, heard notes → coach alerts.

`Session.step()` is the whole pipeline for one frame and has no I/O, so it can be driven by a
live camera, a recorded video, or tests.
"""

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from .coach import Alert, Coach, CoachConfig
from .hands import Hand, HandTracker
from .mapping import HandReading, fretting_hand
from .neck import NeckLocator, NeckPose
from .rig import Rig
from .transcribe import NoteEvent, Transcriber


@dataclass
class FrameResult:
    pose: NeckPose | None
    hand: HandReading | None
    ringing: list[int]
    alerts: list[Alert]


@dataclass
class Session:
    rig: Rig
    tracker: HandTracker | None = None  # None = skip hand tracking (markers only)
    coach_config: CoachConfig = field(default_factory=CoachConfig)

    def __post_init__(self):
        self.neck = NeckLocator(self.rig)
        self.coach = Coach(self.rig.instrument, self.coach_config)
        self.history: list[Alert] = []

    def step(self, frame: np.ndarray, t: float, heard: list[NoteEvent], hands: list[Hand] | None = None) -> FrameResult:
        pose = self.neck.update(frame)
        if hands is None and self.tracker is not None:
            hands = self.tracker.detect(frame, int(t * 1000))
        hand = fretting_hand(hands or [], pose, self.rig.instrument) if pose is not None else None
        alerts = self.coach.update(t, hand, heard)
        self.history.extend(alerts)
        window = self.coach_config.heard_window_s
        ringing = sorted({n.midi for n in heard if n.end >= t - window and n.start <= t})
        return FrameResult(pose, hand, ringing, alerts)


class LiveTranscription(threading.Thread):
    """Re-transcribes the most recent audio every `hop_s` seconds in the background."""

    def __init__(self, audio, transcriber: Transcriber, window_s: float, hop_s: float):
        super().__init__(daemon=True)
        self.audio, self.transcriber = audio, transcriber
        self.window_s, self.hop_s = window_s, hop_s
        self._events: list[NoteEvent] = []
        self._lock = threading.Lock()
        self._running = True

    def run(self) -> None:
        while self._running:
            started = time.monotonic()
            chunk, t0 = self.audio.latest(self.window_s)
            events = self.transcriber.transcribe(chunk, self.audio.sr, t0)
            with self._lock:
                self._events = events
            time.sleep(max(0.0, self.hop_s - (time.monotonic() - started)))

    def events(self) -> list[NoteEvent]:
        with self._lock:
            return list(self._events)

    def stop(self) -> None:
        self._running = False
