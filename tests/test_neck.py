import cv2
import numpy as np
import pytest

from guitra.fretboard import fret_center
from guitra.markers import marker_sheet
from guitra.neck import NeckLocator
from guitra.rig import DEFAULT_RIGS, load_rig
from tests.synth import camera_homography, render_markers

RIG = DEFAULT_RIGS["guitar"]


def test_locates_the_fretboard_from_markers():
    H = camera_homography()
    frame = render_markers(RIG, H)
    pose = NeckLocator(RIG).update(frame)
    assert pose is not None
    assert pose.marker_ids == (0, 1, 2)
    assert pose.reliable
    # fret positions across the whole neck come back within ~2 mm
    for s, f in [(0, 1), (5, 1), (2, 7), (0, 12), (5, 20)]:
        x, y = fret_center(RIG.instrument, s, f)
        px = cv2.perspectiveTransform(np.float32([[[x, y]]]), H)[0]
        back = pose.to_neck(px)[0]
        assert np.hypot(back[0] - x, back[1] - y) < 2.0, (s, f, back, (x, y))


def test_one_marker_works_but_is_flagged_unreliable():
    frame = render_markers(RIG, camera_homography(), only={0})
    pose = NeckLocator(RIG).update(frame)
    assert pose is not None and pose.marker_ids == (0,)
    assert not pose.reliable


def test_holds_the_last_pose_briefly_when_markers_disappear():
    loc = NeckLocator(RIG, hold_frames=2)
    loc.update(render_markers(RIG, camera_homography()))
    blank = np.full((720, 1280, 3), 170, np.uint8)
    assert loc.update(blank).age_frames == 1
    assert loc.update(blank).age_frames == 2
    assert loc.update(blank) is None


def test_printed_marker_sheet_is_detectable():
    sheet = marker_sheet(RIG)
    small = cv2.resize(sheet, None, fx=0.5, fy=0.5)
    detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))
    _, ids, _ = detector.detectMarkers(small)
    assert sorted(ids.ravel().tolist()) == [0, 1, 2]


@pytest.mark.parametrize("name, strings", [("guitar", 6), ("bass", 4)])
def test_example_rigs_load(name, strings):
    rig = load_rig(f"examples/{name}.toml")
    assert rig.instrument.strings == strings
    assert [m.id for m in rig.markers] == [0, 1, 2]
