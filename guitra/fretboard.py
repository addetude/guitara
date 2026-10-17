"""Fretboard geometry in "neck coordinates" (millimetres).

    x: distance along the neck from the nut towards the body (0 = nut)
    y: distance across the neck from its centreline; negative = towards the lowest string

Fret n sits at x = L * (1 - 2^(-n/12)) (the rule of 18: each fret is 1/17.817 closer).
Strings fan out linearly from the nut spread to the bridge spread.
"""

from dataclasses import dataclass

from .instruments import Instrument


@dataclass(frozen=True)
class FretPosition:
    """Where a point on the neck falls, in player terms."""

    string: int  # 0 = lowest string
    fret: int  # fret being pressed (1 = first fret); 0 = behind the nut
    # 0 = right behind the previous fret wire, 1 = on the fret wire you're pressing.
    # Players should be close to the wire (~0.6–0.95) to avoid buzzing.
    depth_in_fret: float
    string_offset_mm: float  # how far off the string centre the point is
    on_neck: bool


def fret_x(inst: Instrument, n: float) -> float:
    return inst.scale_length_mm * (1 - 2 ** (-n / 12))


def string_spread(inst: Instrument, x: float) -> float:
    t = max(0.0, min(1.0, x / inst.scale_length_mm))
    return inst.nut_width_mm + (inst.bridge_spread_mm - inst.nut_width_mm) * t


def string_y(inst: Instrument, string: int, x: float) -> float:
    if inst.strings == 1:
        return 0.0
    return (string / (inst.strings - 1) - 0.5) * string_spread(inst, x)


def string_spacing(inst: Instrument, x: float) -> float:
    return string_spread(inst, x) / max(1, inst.strings - 1)


def neck_half_width(inst: Instrument, x: float) -> float:
    """Half the playable width: outer strings plus half a string spacing of margin."""
    return string_spread(inst, x) / 2 + string_spacing(inst, x) / 2


def fret_center(inst: Instrument, string: int, fret: int, depth: float = 0.8) -> tuple[float, float]:
    """A good fingertip spot for (string, fret): `depth` of the way towards the fret wire."""
    x0 = fret_x(inst, fret - 1)
    x1 = fret_x(inst, fret)
    x = x0 + depth * (x1 - x0)
    return x, string_y(inst, string, x)


def locate(inst: Instrument, x: float, y: float) -> FretPosition:
    """Maps a point in neck coordinates to the nearest string and the fret space it's in."""
    # nearest string
    rel = (y / string_spread(inst, x) + 0.5) * (inst.strings - 1)
    string = int(round(min(inst.strings - 1, max(0, rel))))
    offset = y - string_y(inst, string, x)

    if x <= 0:
        fret, depth = 0, 0.0
    else:
        fret = 1
        while fret < inst.frets and fret_x(inst, fret) < x:
            fret += 1
        x0, x1 = fret_x(inst, fret - 1), fret_x(inst, fret)
        depth = (x - x0) / (x1 - x0)

    on_neck = 0 < x <= fret_x(inst, inst.frets) and abs(y) <= neck_half_width(inst, x)
    return FretPosition(string, fret, depth, offset, on_neck)
