"""Instrument definitions: tuning and neck dimensions.

Strings are numbered the way the coach talks about them to a player: index 0 is the
*lowest-pitched* string (low E on guitar), and `string_label` gives the conventional
name ("6th string (low E)").
"""

from dataclasses import dataclass

from .notes import midi_to_name, name_to_midi


@dataclass(frozen=True)
class Instrument:
    name: str
    tuning: tuple[int, ...]  # MIDI note of each open string, lowest first
    scale_length_mm: float  # nut to bridge saddle
    nut_width_mm: float  # outer string spread at the nut
    bridge_spread_mm: float  # outer string spread at the bridge
    frets: int

    @property
    def strings(self) -> int:
        return len(self.tuning)

    def note_at(self, string: int, fret: int) -> int:
        return self.tuning[string] + fret

    def string_label(self, string: int) -> str:
        number = self.strings - string  # low E on a guitar is the 6th string
        return f"string {number} ({midi_to_name(self.tuning[string])})"


def _tuning(*names: str) -> tuple[int, ...]:
    return tuple(name_to_midi(n) for n in names)


GUITAR = Instrument(
    name="guitar",
    tuning=_tuning("E2", "A2", "D3", "G3", "B3", "E4"),
    scale_length_mm=648.0,  # 25.5" (Fender); a Gibson is ~628 mm
    nut_width_mm=35.0,  # outer strings, not the full 42 mm nut
    bridge_spread_mm=52.0,
    frets=22,
)

BASS = Instrument(
    name="bass",
    tuning=_tuning("E1", "A1", "D2", "G2"),
    scale_length_mm=864.0,  # 34" long-scale
    nut_width_mm=29.0,
    bridge_spread_mm=57.0,
    frets=20,
)

INSTRUMENTS = {i.name: i for i in (GUITAR, BASS)}
