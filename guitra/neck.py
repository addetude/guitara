"""Finds the fretboard in a camera frame from ArUco markers (OpenCV) and maps between
image pixels and neck coordinates with a homography.

Every visible marker contributes its 4 corners as point pairs (pixel ↔ millimetre). Two or
more markers spread along the neck give a stable fit; one marker works but extrapolates
poorly far from it. The fit is smoothed over frames and held briefly if markers drop out
(e.g. your hand passes in front of one).
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .rig import MarkerSpec, Rig


def marker_corners_mm(m: MarkerSpec, handed: str) -> np.ndarray:
    """Neck-coordinate positions of a marker's corners, in OpenCV's order (TL, TR, BR, BL).

    Convention: stick each marker upright as seen from the front of the guitar, top edge
    towards the low strings. Seen from the front, a right-handed guitar's headstock points
    to the camera's right, so the marker's "right" points towards the nut (−x).
    """
    h = m.size_mm / 2
    right = -1.0 if handed == "right" else 1.0  # marker's right edge direction along x
    u = np.array([right, 0.0])  # marker right, in neck coords
    v = np.array([0.0, 1.0])  # marker down = towards the high strings
    c = np.array([m.x_mm, m.y_mm])
    return np.array([c - h * u - h * v, c + h * u - h * v, c + h * u + h * v, c - h * u + h * v], dtype=np.float32)


@dataclass
class NeckPose:
    image_to_neck: np.ndarray  # 3x3 homography, pixels → mm
    neck_to_image: np.ndarray
    marker_ids: tuple[int, ...]  # markers used for this fit
    reprojection_error_mm: float
    age_frames: int = 0  # 0 = fitted this frame; >0 = held from an earlier frame

    def to_neck(self, pts_px: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_px, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.image_to_neck).reshape(-1, 2)

    def to_image(self, pts_mm: np.ndarray) -> np.ndarray:
        pts = np.asarray(pts_mm, dtype=np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(pts, self.neck_to_image).reshape(-1, 2)

    @property
    def reliable(self) -> bool:
        return len(self.marker_ids) >= 2 and self.reprojection_error_mm < 3.0


class NeckLocator:
    def __init__(self, rig: Rig, smoothing: float = 0.5, hold_frames: int = 15):
        self.rig = rig
        dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, rig.aruco_dictionary))
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
        self.detector = cv2.aruco.ArucoDetector(dictionary, params)
        self.smoothing = smoothing
        self.hold_frames = hold_frames
        self.pose: NeckPose | None = None
        self.last_detections: list[tuple[int, np.ndarray]] = []

    def update(self, frame_bgr: np.ndarray) -> NeckPose | None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        corners, ids, _ = self.detector.detectMarkers(gray)
        self.last_detections = []
        img_pts, neck_pts, used = [], [], []
        if ids is not None:
            for c, marker_id in zip(corners, ids.ravel()):
                spec = self.rig.marker(int(marker_id))
                px = c.reshape(4, 2)
                self.last_detections.append((int(marker_id), px))
                if spec is None:
                    continue
                img_pts.append(px)
                neck_pts.append(marker_corners_mm(spec, self.rig.handed))
                used.append(int(marker_id))

        if not used:
            return self._hold()

        src = np.concatenate(img_pts).astype(np.float32)
        dst = np.concatenate(neck_pts).astype(np.float32)
        H, _ = cv2.findHomography(src, dst, 0)
        if H is None:
            return self._hold()
        H = H / H[2, 2]
        if self.pose is not None and self.pose.age_frames == 0 and set(used) == set(self.pose.marker_ids):
            H = self.smoothing * self.pose.image_to_neck + (1 - self.smoothing) * H  # damp jitter
            H = H / H[2, 2]
        proj = cv2.perspectiveTransform(src.reshape(-1, 1, 2), H).reshape(-1, 2)
        err = float(np.mean(np.linalg.norm(proj - dst, axis=1)))
        self.pose = NeckPose(H, np.linalg.inv(H), tuple(sorted(used)), err)
        return self.pose

    def _hold(self) -> NeckPose | None:
        if self.pose is None:
            return None
        self.pose.age_frames += 1
        if self.pose.age_frames > self.hold_frames:
            self.pose = None
        return self.pose
