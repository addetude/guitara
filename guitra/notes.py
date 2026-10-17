"""Pitch helpers. MIDI 60 = C4 (middle C)."""

import math

NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def midi_to_name(midi: int) -> str:
    return f"{NAMES[midi % 12]}{midi // 12 - 1}"


def name_to_midi(name: str) -> int:
    """'E2' -> 40, 'F#3' -> 54, 'Bb1' -> 34."""
    letter, rest = name[0].upper(), name[1:]
    pc = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}[letter]
    while rest and rest[0] in "#b":
        pc += 1 if rest[0] == "#" else -1
        rest = rest[1:]
    return pc + (int(rest) + 1) * 12


def midi_to_hz(midi: float) -> float:
    return 440.0 * 2 ** ((midi - 69) / 12)


def hz_to_midi(hz: float) -> float:
    return 69 + 12 * math.log2(hz / 440.0)
