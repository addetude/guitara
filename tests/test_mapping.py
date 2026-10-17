from guitra.app import Session
from guitra.rig import DEFAULT_RIGS
from tests.synth import camera_homography, make_hand, render_markers, tips_at

RIG = DEFAULT_RIGS["guitar"]
INST = RIG.instrument


def test_fingertips_map_to_string_and_fret_through_the_camera():
    H = camera_homography()
    frame = render_markers(RIG, H)
    # C major (x32010): ring on A string fret 3, middle on D fret 2, index on B fret 1
    fretting = make_hand(tips_at(INST, H, {"index": (4, 1), "middle": (2, 2), "ring": (1, 3), "pinky": (0, 4)}))
    strumming = make_hand({"index": (300, 650), "middle": (310, 660), "ring": (320, 670), "pinky": (330, 680)})
    result = Session(RIG).step(frame, 0.0, [], hands=[strumming, fretting])
    assert result.hand is not None and result.hand.hand is fretting
    got = {f.finger: (f.pos.string, f.pos.fret) for f in result.hand.on_neck}
    assert got == {"index": (4, 1), "middle": (2, 2), "ring": (1, 3), "pinky": (0, 4)}


def test_no_markers_means_no_hand_reading():
    import numpy as np

    frame = np.full((720, 1280, 3), 170, np.uint8)
    hand = make_hand({"index": (600, 300)})
    assert Session(RIG).step(frame, 0.0, [], hands=[hand]).hand is None
