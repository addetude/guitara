"""Your rig: which instrument, which hand you fret with, and where the ArUco markers are stuck.

Loaded from a TOML file (see `examples/`). Marker positions are in neck coordinates:
x = millimetres from the nut towards the body (negative = on the headstock),
y = millimetres from the neck's centre line (negative = towards the lowest string).
"""

import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

from .fretboard import fret_x
from .instruments import BASS, GUITAR, INSTRUMENTS, Instrument


@dataclass(frozen=True)
class MarkerSpec:
    id: int
    x_mm: float  # marker centre
    y_mm: float
    size_mm: float  # black square edge length, as printed


@dataclass(frozen=True)
class Rig:
    instrument: Instrument
    # "right" = right-handed player (frets with the left hand, headstock on the camera's right)
    handed: str = "right"
    markers: tuple[MarkerSpec, ...] = field(default_factory=tuple)
    aruco_dictionary: str = "DICT_4X4_50"

    def marker(self, marker_id: int) -> MarkerSpec | None:
        return next((m for m in self.markers if m.id == marker_id), None)


def default_markers(inst: Instrument) -> tuple[MarkerSpec, ...]:
    """A starting layout: one marker on the headstock, two on the body either side of the neck.
    Measure your own guitar and put the real numbers in your setup file."""
    body_x = fret_x(inst, inst.frets) + 25
    side = inst.bridge_spread_mm / 2 + 30
    return (
        MarkerSpec(0, -40.0, 0.0, 25.0),
        MarkerSpec(1, body_x, side, 30.0),
        MarkerSpec(2, body_x, -side, 30.0),
    )


DEFAULT_RIGS = {
    "guitar": Rig(GUITAR, markers=default_markers(GUITAR)),
    "bass": Rig(BASS, markers=default_markers(BASS)),
}


def load_rig(path: str | Path) -> Rig:
    data = tomllib.loads(Path(path).read_text())
    base = INSTRUMENTS[data.get("instrument", "guitar")]
    overrides = {
        k: data[k]
        for k in ("scale_length_mm", "nut_width_mm", "bridge_spread_mm", "frets")
        if k in data
    }
    if "tuning" in data:
        from .notes import name_to_midi

        overrides["tuning"] = tuple(name_to_midi(n) for n in data["tuning"])
    inst = replace(base, **overrides)
    markers = tuple(
        MarkerSpec(int(m["id"]), float(m["x_mm"]), float(m["y_mm"]), float(m["size_mm"]))
        for m in data.get("markers", [])
    ) or default_markers(inst)
    handed = data.get("handed", "right")
    if handed not in ("right", "left"):
        raise ValueError(f"handed must be 'right' or 'left', got {handed!r}")
    return Rig(inst, handed, markers, data.get("aruco_dictionary", "DICT_4X4_50"))
