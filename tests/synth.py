"""Synthetic test fixtures: a camera frame with ArUco markers in perspective, fake hands, notes."""

import cv2
import numpy as np

from guitra.fretboard import fret_center
from guitra.hands import Hand
from guitra.neck import NeckPose, marker_corners_mm
from guitra.rig import Rig

FRAME_W, FRAME_H = 1280, 720


def camera_homography() -> np.ndarray:
    """Neck mm → image px for a right-handed guitar seen from the front, slightly in perspective:
    nut on the right, body on the left, low strings at the top."""
    neck = np.float32([[-80, -90], [560, -90], [560, 90], [-80, 90]])
    image = np.float32([[1230, 250], [80, 210], [70, 520], [1220, 470]])
    return cv2.getPerspectiveTransform(neck, image)


def render_markers(rig: Rig, H: np.ndarray, only: set[int] | None = None) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, rig.aruco_dictionary))
    frame = np.full((FRAME_H, FRAME_W, 3), 170, np.uint8)  # grey "guitar body"
    for m in rig.markers:
        if only is not None and m.id not in only:
            continue
        side, quiet = 200, 50
        tile = np.full((side + 2 * quiet, side + 2 * quiet), 255, np.uint8)
        tile[quiet:quiet + side, quiet:quiet + side] = cv2.aruco.generateImageMarker(dictionary, m.id, side)
        src = np.float32([[quiet, quiet], [quiet + side, quiet], [quiet + side, quiet + side], [quiet, quiet + side]])
        dst = cv2.perspectiveTransform(marker_corners_mm(m, rig.handed).reshape(-1, 1, 2), H).reshape(-1, 2)
        M = cv2.getPerspectiveTransform(src, dst.astype(np.float32))
        warped = cv2.warpPerspective(tile, M, (FRAME_W, FRAME_H), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full_like(tile, 255), M, (FRAME_W, FRAME_H)) > 0
        frame[mask] = warped[mask][:, None]
    return frame


def identity_pose() -> NeckPose:
    """A 'camera' whose pixels are neck millimetres — handy for coach tests."""
    return NeckPose(np.eye(3), np.eye(3), (0, 1), 0.0)


def make_hand(tips_px: dict[str, tuple[float, float]], z: dict[str, float] | None = None, size_px: float = 80) -> Hand:
    """A hand whose fingertips are at the given pixel positions (other landmarks roughly placed)."""
    tip_idx = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
    pts = np.zeros((21, 3), np.float32)
    cx = np.mean([p[0] for p in tips_px.values()])
    cy = np.mean([p[1] for p in tips_px.values()])
    for i in range(21):
        pts[i] = (cx, cy + size_px * 0.6, 0)
    pts[0] = (cx, cy + size_px * 1.5, 0)  # wrist
    pts[9] = (cx, cy + size_px * 0.5, 0)  # middle knuckle → hand size = size_px
    for name, (x, y) in tips_px.items():
        pts[tip_idx[name]] = (x, y, (z or {}).get(name, 0.0))
    return Hand(pts, "Right", 0.95)


def tips_at(inst, H, placements: dict[str, tuple[int, int] | tuple[float, float, str]], depth=0.8):
    """Fingertip pixel positions for {finger: (string, fret)}; raw (x_mm, y_mm, 'mm') also allowed."""
    out = {}
    for finger, spec in placements.items():
        if len(spec) == 3:
            x, y = spec[0], spec[1]
        else:
            x, y = fret_center(inst, spec[0], spec[1], depth)
        out[finger] = tuple(cv2.perspectiveTransform(np.float32([[[x, y]]]), H)[0, 0])
    return out
