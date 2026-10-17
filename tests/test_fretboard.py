import pytest

from guitra.fretboard import fret_center, fret_x, locate, neck_half_width
from guitra.instruments import BASS, GUITAR
from guitra.notes import midi_to_name, name_to_midi


def test_fret_positions_follow_the_twelfth_root_of_two():
    assert fret_x(GUITAR, 12) == pytest.approx(GUITAR.scale_length_mm / 2)
    assert fret_x(GUITAR, 24) == pytest.approx(GUITAR.scale_length_mm * 0.75)
    # each fret space is ~5.6% shorter than the previous one
    first = fret_x(GUITAR, 1) - fret_x(GUITAR, 0)
    second = fret_x(GUITAR, 2) - fret_x(GUITAR, 1)
    assert second / first == pytest.approx(2 ** (-1 / 12))


@pytest.mark.parametrize("inst", [GUITAR, BASS])
def test_every_string_and_fret_round_trips(inst):
    for s in range(inst.strings):
        for f in range(1, inst.frets + 1):
            x, y = fret_center(inst, s, f)
            p = locate(inst, x, y)
            assert (p.string, p.fret) == (s, f)
            assert p.on_neck
            assert 0.7 < p.depth_in_fret < 0.9


def test_points_off_the_neck():
    assert not locate(GUITAR, 100, neck_half_width(GUITAR, 100) + 5).on_neck
    assert not locate(GUITAR, -10, 0).on_neck  # headstock side of the nut


def test_depth_in_fret_measures_distance_from_the_fret_wire():
    x0, x1 = fret_x(GUITAR, 4), fret_x(GUITAR, 5)
    assert locate(GUITAR, x0 + 0.1 * (x1 - x0), 0).depth_in_fret == pytest.approx(0.1)
    assert locate(GUITAR, x0 + 0.95 * (x1 - x0), 0).fret == 5


def test_notes_and_labels():
    assert GUITAR.note_at(0, 5) == GUITAR.tuning[1]  # 5th fret of low E = open A
    assert GUITAR.note_at(3, 4) == GUITAR.tuning[4]  # 4th fret of G = open B
    assert BASS.tuning[0] == name_to_midi("E1") == 28
    assert midi_to_name(40) == "E2"
    assert GUITAR.string_label(0) == "string 6 (E2)"
    assert BASS.string_label(3) == "string 1 (G2)"
