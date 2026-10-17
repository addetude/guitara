"""The coach: compares what your fretting hand is doing (camera) with what came out of the amp
(audio) and raises alerts. Each check needs to see the problem for a moment before it fires,
and won't repeat the same alert for a few seconds, so it doesn't nag.

Checks:
  muted-string   You're fretting a string inside a strummed chord, but its note isn't ringing.
  wrong-fret     A note was heard that's a semitone or two off from what your fingers show.
  far-from-fret  A finger is pressing a note from far behind the fret wire (buzz risk).
  lazy-pinky     Ring finger stretching to the 4th fret of the position instead of using the pinky.
  flying-pinky   Pinky drifting far away from the neck while the other fingers play.
"""

from dataclasses import dataclass, field

from .fretboard import fret_x, neck_half_width
from .instruments import Instrument
from .mapping import FingerPlacement, HandReading
from .notes import midi_to_name
from .transcribe import NoteEvent


@dataclass(frozen=True)
class Alert:
    kind: str
    key: str  # identifies "the same problem" for cooldowns
    message: str
    t: float


@dataclass
class CoachConfig:
    persist_s: float = 0.35  # how long a problem must be seen before alerting
    cooldown_s: float = 5.0  # don't repeat the same alert within this time
    heard_window_s: float = 0.6  # notes heard this recently count as "ringing now"
    far_from_fret_depth: float = 0.4  # fraction of the fret space; < this = too far back
    flying_pinky_mm: float = 25.0  # pinky further than this outside the neck edge
    flying_pinky_depth: float = 0.45  # or lifted towards the camera by this × hand size (MediaPipe z)
    flying_pinky_persist_s: float = 1.0


@dataclass
class Coach:
    inst: Instrument
    config: CoachConfig = field(default_factory=CoachConfig)
    _first_seen: dict[str, float] = field(default_factory=dict)
    _last_fired: dict[str, float] = field(default_factory=dict)

    def update(self, t: float, hand: HandReading | None, heard: list[NoteEvent]) -> list[Alert]:
        """Call once per camera frame. `heard` = notes from the transcriber (any recent ones)."""
        ringing = {n.midi for n in heard if n.end >= t - self.config.heard_window_s and n.start <= t}
        candidates: dict[str, tuple[str, str, float]] = {}  # key → (kind, message, persist)
        if hand is not None:
            fretted = self._fretted_strings(hand)
            for c in (
                self._muted_strings(fretted, ringing, hand),
                self._wrong_frets(fretted, ringing),
                self._far_from_fret(fretted, ringing),
                self._lazy_pinky(hand, fretted, ringing),
                self._flying_pinky(hand),
            ):
                candidates.update(c)
        return self._debounce(t, candidates)

    # ---- what the hand says ----------------------------------------------------------------

    def _fretted_strings(self, hand: HandReading) -> dict[int, FingerPlacement]:
        """For each string with a finger on it, the finger at the highest fret (that's the note
        that sounds when several fingers are on one string)."""
        out: dict[int, FingerPlacement] = {}
        for f in hand.on_neck:
            cur = out.get(f.pos.string)
            if cur is None or f.pos.fret > cur.pos.fret:
                out[f.pos.string] = f
        return out

    def _expected(self, f: FingerPlacement) -> int:
        return self.inst.note_at(f.pos.string, f.pos.fret)

    def _heard(self, midi: int, ringing: set[int], octave_ok: bool = True) -> bool:
        # Pitch trackers sometimes report the wrong octave, so single-note checks accept it.
        # Chords often contain the same note in two octaves, so the muted-string check doesn't.
        if midi in ringing:
            return True
        return octave_ok and (midi + 12 in ringing or midi - 12 in ringing)

    # ---- checks -----------------------------------------------------------------------------

    def _muted_strings(self, fretted: dict[int, FingerPlacement], ringing: set[int], hand: HandReading):
        out = {}
        if len(ringing) < 2:  # only judge strummed chords: a single picked note says nothing
            return out
        lo, hi = min(ringing), max(ringing)
        for s, f in fretted.items():
            note = self._expected(f)
            if self._heard(note, ringing, octave_ok=False) or not (lo < note < hi):
                continue  # ringing, or outside the strummed range (maybe you didn't strum it)
            label = self.inst.string_label(s)
            msg = (
                f"{label[0].upper() + label[1:]} at fret {f.pos.fret} isn't ringing "
                f"({midi_to_name(note)} missing). Press right behind the fret on your fingertip"
            )
            neighbour = next((g for g in hand.on_neck if abs(g.pos.string - s) == 1 and g.finger != f.finger), None)
            if neighbour:
                msg += f", and arch finger {neighbour.number} so it doesn't lean on this string"
            out[f"muted:{s}"] = ("muted-string", msg + ".", self.config.persist_s)
        return out

    def _wrong_frets(self, fretted: dict[int, FingerPlacement], ringing: set[int]):
        out = {}
        expected = {self._expected(f): f for f in fretted.values()}
        open_notes = {self.inst.tuning[s] for s in range(self.inst.strings) if s not in fretted}
        for h in ringing:
            if h in expected or h in open_notes or h - 12 in expected or h + 12 in expected:
                continue
            near = [(abs(h - e), e, f) for e, f in expected.items() if 0 < abs(h - e) <= 2]
            if not near:
                continue
            _, e, f = min(near, key=lambda n: n[0])
            out[f"wrong:{f.pos.string}:{f.pos.fret}"] = (
                "wrong-fret",
                f"Heard {midi_to_name(h)}, but finger {f.number} looks like it's on "
                f"{self.inst.string_label(f.pos.string)}, fret {f.pos.fret} ({midi_to_name(e)}). "
                "Check you're on the right fret.",
                self.config.persist_s,
            )
        return out

    def _far_from_fret(self, fretted: dict[int, FingerPlacement], ringing: set[int]):
        out = {}
        for s, f in fretted.items():
            # only judge fingers that are actually sounding a note (audio confirms the press)
            if f.pos.depth_in_fret < self.config.far_from_fret_depth and self._heard(self._expected(f), ringing):
                out[f"far:{f.finger}"] = (
                    "far-from-fret",
                    f"Finger {f.number} is pressing far behind fret {f.pos.fret}. "
                    "Move it up close to the fret wire for a cleaner note with less pressure.",
                    self.config.persist_s,
                )
        return out

    def _lazy_pinky(self, hand: HandReading, fretted: dict[int, FingerPlacement], ringing: set[int]):
        """One finger per fret: with the index on fret n, fret n+3 belongs to the pinky."""
        index, ring, pinky = hand.fingers["index"], hand.fingers["ring"], hand.fingers["pinky"]
        if not (index.on_string and ring.on_string):
            return {}
        stretch = ring.pos.fret - index.pos.fret
        ring_sounding = fretted.get(ring.pos.string) is ring and self._heard(self._expected(ring), ringing)
        pinky_there = pinky.on_string and pinky.pos.fret >= ring.pos.fret
        if stretch >= 3 and ring_sounding and not pinky_there:
            return {
                "lazy-pinky": (
                    "lazy-pinky",
                    f"Your ring finger is stretching to fret {ring.pos.fret}. With your index on fret "
                    f"{index.pos.fret}, that fret belongs to your pinky (one finger per fret).",
                    self.config.persist_s,
                )
            }
        return {}

    def _flying_pinky(self, hand: HandReading):
        pinky = hand.fingers["pinky"]
        others = [hand.fingers[n] for n in ("index", "middle", "ring")]
        if not any(f.on_string for f in others) or pinky.on_string:
            return {}
        # (a) far outside the neck in the fretboard plane
        edge = neck_half_width(self.inst, max(0.0, pinky.x_mm))
        outside_mm = abs(pinky.y_mm) - edge
        behind_nut = pinky.x_mm < fret_x(self.inst, 0) - self.config.flying_pinky_mm
        # (b) lifted towards the camera: MediaPipe z is negative towards the camera
        lm = hand.hand.landmarks_px
        others_z = sum(lm[t, 2] for t in (8, 12, 16)) / 3
        lifted = (others_z - lm[20, 2]) > self.config.flying_pinky_depth * hand.hand_size_px
        if outside_mm > self.config.flying_pinky_mm or behind_nut or lifted:
            return {
                "flying-pinky": (
                    "flying-pinky",
                    "Your pinky is flying away from the neck. Keep it hovering just above the strings "
                    "so it's ready for the next note.",
                    self.config.flying_pinky_persist_s,
                )
            }
        return {}

    # ---- timing -----------------------------------------------------------------------------

    def _debounce(self, t: float, candidates: dict[str, tuple[str, str, float]]) -> list[Alert]:
        for key in list(self._first_seen):
            if key not in candidates:
                del self._first_seen[key]  # problem went away: start over next time
        alerts = []
        for key, (kind, message, persist) in candidates.items():
            first = self._first_seen.setdefault(key, t)
            if t - first < persist:
                continue
            if t - self._last_fired.get(key, -1e9) < self.config.cooldown_s:
                continue
            self._last_fired[key] = t
            alerts.append(Alert(kind, key, message, t))
        return alerts
