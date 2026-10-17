import numpy as np

from guitra.coach import Coach
from guitra.fretboard import fret_center
from guitra.mapping import read_hand
from guitra.rig import DEFAULT_RIGS
from guitra.transcribe import NoteEvent
from tests.synth import identity_pose, make_hand, tips_at

INST = DEFAULT_RIGS["guitar"].instrument
POSE = identity_pose()
I = np.eye(3)


def hand(placements, depth=0.8, z=None):
    return read_hand(make_hand(tips_at(INST, I, placements, depth), z=z), POSE, INST)


def notes(*midis, t=0.0):
    return [NoteEvent(m, t - 1.0, t + 10.0, 0.8) for m in midis]


def run(coach, h, heard, times):
    alerts = []
    for t in times:
        alerts += coach.update(t, h, heard)
    return alerts


C_MAJOR = {"index": (4, 1), "middle": (2, 2), "ring": (1, 3)}  # x32010
C_MAJOR_NOTES = (48, 52, 55, 60, 64)  # C3 E3 G3 C4 E4


def test_clean_chord_gives_no_alerts():
    assert run(Coach(INST), hand(C_MAJOR), notes(*C_MAJOR_NOTES), [0, 0.5, 1.0]) == []


def test_muted_string_in_a_strummed_chord():
    heard = notes(48, 55, 60, 64)  # E3 on the D string is missing
    alerts = run(Coach(INST), hand(C_MAJOR), heard, [0, 0.2, 0.4])
    assert [a.kind for a in alerts] == ["muted-string"]
    assert alerts[0].message.startswith("String 4 (D3) at fret 2")
    assert "arch finger" in alerts[0].message  # a neighbouring finger may be leaning on it


def test_alerts_need_to_persist_and_then_cool_down():
    coach = Coach(INST)
    h, heard = hand(C_MAJOR), notes(48, 55, 60, 64)
    assert coach.update(0.0, h, heard) == []  # seen once: not yet
    assert len(coach.update(0.4, h, heard)) == 1  # persisted
    assert coach.update(1.0, h, heard) == []  # cooling down
    assert len(coach.update(6.0, h, heard)) == 1


def test_single_picked_notes_never_count_as_muted():
    assert run(Coach(INST), hand(C_MAJOR), notes(60), [0, 0.5]) == []


def test_wrong_fret():
    alerts = run(Coach(INST), hand({"index": (4, 1)}), notes(61), [0, 0.5])  # C#4 instead of C4
    assert [a.kind for a in alerts] == ["wrong-fret"]
    assert "Heard C#4" in alerts[0].message


def test_far_from_fret_only_when_the_note_actually_sounds():
    far = hand({"index": (4, 1)}, depth=0.2)
    assert [a.kind for a in run(Coach(INST), far, notes(60), [0, 0.5])] == ["far-from-fret"]
    assert run(Coach(INST), far, [], [0, 0.5]) == []  # hovering, not pressing
    assert run(Coach(INST), hand({"index": (4, 1)}, depth=0.85), notes(60), [0, 0.5]) == []


def test_lazy_pinky_ring_finger_stretching_to_the_fourth_fret():
    # index on fret 5, ring stretching to fret 8 on the same string, pinky floating
    h = hand({"index": (0, 5), "ring": (0, 8), "pinky": (90.0 * 0 + 330, -60, "mm")})
    alerts = run(Coach(INST), h, notes(INST.note_at(0, 8)), [0, 0.5])
    assert "lazy-pinky" in [a.kind for a in alerts]
    # using the pinky for fret 8 is fine
    good = hand({"index": (0, 5), "ring": (0, 7), "pinky": (0, 8)})
    assert "lazy-pinky" not in [a.kind for a in run(Coach(INST), good, notes(INST.note_at(0, 8)), [0, 0.5])]


def test_flying_pinky_off_the_neck():
    x, _ = fret_center(INST, 0, 5)
    h = hand({"index": (3, 5), "middle": (3, 6), "pinky": (x + 40, 80, "mm")})
    coach = Coach(INST)
    assert run(coach, h, [], [0, 0.5]) == []  # needs a full second
    assert [a.kind for a in run(coach, h, [], [1.1])] == ["flying-pinky"]


def test_flying_pinky_lifted_towards_the_camera():
    x, y = fret_center(INST, 1, 8)
    lifted = hand({"index": (3, 5), "middle": (3, 6), "ring": (3, 7), "pinky": (x, y + 4, "mm")}, z={"pinky": -60})
    # pinky is over the neck but between strings and lifted: MediaPipe z says it's up off the board
    assert "flying-pinky" in [a.kind for a in run(Coach(INST), lifted, [], [0, 1.1])]
