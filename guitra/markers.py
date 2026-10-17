"""Printable ArUco marker sheet (OpenCV), sized in millimetres for an A4/Letter page at 300 dpi."""

import cv2
import numpy as np

from .rig import Rig

DPI = 300
PAGE_MM = (210, 279)  # fits both A4 and US Letter


def mm_to_px(mm: float, dpi: int = DPI) -> int:
    return int(round(mm / 25.4 * dpi))


def marker_sheet(rig: Rig, dpi: int = DPI) -> np.ndarray:
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, rig.aruco_dictionary))
    page = np.full((mm_to_px(PAGE_MM[1], dpi), mm_to_px(PAGE_MM[0], dpi)), 255, np.uint8)
    margin = mm_to_px(15, dpi)
    x, y, row_h = margin, margin + mm_to_px(12, dpi), 0
    cv2.putText(page, "guitra markers - print at 100% scale (no 'fit to page'), check the size with a ruler",
                (margin, margin), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
    for m in rig.markers:
        side = mm_to_px(m.size_mm, dpi)
        quiet = mm_to_px(max(4.0, m.size_mm * 0.2), dpi)  # white border the detector needs
        cell = side + 2 * quiet
        if x + cell > page.shape[1] - margin:
            x, y, row_h = margin, y + row_h + mm_to_px(16, dpi), 0
        img = cv2.aruco.generateImageMarker(dictionary, m.id, side)
        page[y + quiet : y + quiet + side, x + quiet : x + quiet + side] = img
        cv2.rectangle(page, (x, y), (x + cell, y + cell), 200, 1)  # cut line
        cv2.putText(page, f"id {m.id}  {m.size_mm:g} mm", (x, y + cell + mm_to_px(5, dpi)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, 0, 2)
        cv2.putText(page, "top edge -> low strings", (x, y + cell + mm_to_px(10, dpi)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, 80, 2)
        x += cell + mm_to_px(10, dpi)
        row_h = max(row_h, cell + mm_to_px(12, dpi))
    return page
