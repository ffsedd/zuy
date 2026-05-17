from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

Point = Tuple[int, int]


# =====================================================
# CONFIG
# =====================================================

IMG_OPT = "optical.png"
IMG_SEM = "sem.png"

ALPHA = 0.5
RADIUS = 8


# =====================================================
# DATA MODEL
# =====================================================


@dataclass
class ControlPoints:
    opt: List[Point]
    sem: List[Point]

    def valid(self) -> bool:
        return len(self.opt) >= 4 and len(self.opt) == len(self.sem)


cp = ControlPoints([], [])

drag_idx: Optional[int] = None
drag_mode: Optional[str] = None  # "opt" or "sem"


# =====================================================
# UTIL
# =====================================================


def nearest(pt: Point, pts: List[Point]) -> Optional[int]:
    if not pts:
        return None

    d = [np.hypot(p[0] - pt[0], p[1] - pt[1]) for p in pts]
    i = int(np.argmin(d))

    if d[i] < RADIUS * 2:
        return i
    return None


def draw_points(img: np.ndarray, pts: List[Point]) -> np.ndarray:
    out = img.copy()

    for i, (x, y) in enumerate(pts):
        cv2.circle(out, (x, y), RADIUS, (0, 0, 255), 2)
        cv2.putText(out, str(i + 1), (x + 6, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return out


# =====================================================
# HOMOGRAPHY
# =====================================================


def compute_overlay(img_opt, img_sem):

    if not cp.valid():
        return img_opt

    src = np.array(cp.sem, np.float32)
    dst = np.array(cp.opt, np.float32)

    H, _ = cv2.findHomography(src, dst, cv2.RANSAC)

    warped = cv2.warpPerspective(img_sem, H, (img_opt.shape[1], img_opt.shape[0]))

    return cv2.addWeighted(img_opt, 1 - ALPHA, warped, ALPHA, 0)


# =====================================================
# MOUSE HANDLERS
# =====================================================


def mouse_opt(event, x, y, flags, param):
    global drag_idx, drag_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        idx = nearest((x, y), cp.opt)

        if idx is not None:
            drag_idx = idx
            drag_mode = "opt"
        else:
            cp.opt.append((x, y))

    elif event == cv2.EVENT_MOUSEMOVE:
        if drag_mode == "opt" and drag_idx is not None:
            cp.opt[drag_idx] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drag_idx = None
        drag_mode = None


def mouse_sem(event, x, y, flags, param):
    global drag_idx, drag_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        idx = nearest((x, y), cp.sem)

        if idx is not None:
            drag_idx = idx
            drag_mode = "sem"
        else:
            cp.sem.append((x, y))

    elif event == cv2.EVENT_MOUSEMOVE:
        if drag_mode == "sem" and drag_idx is not None:
            cp.sem[drag_idx] = (x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drag_idx = None
        drag_mode = None


# =====================================================
# MAIN GUI LOOP
# =====================================================


def main():

    img_opt = cv2.imread(IMG_OPT)
    img_sem = cv2.imread(IMG_SEM, cv2.IMREAD_GRAYSCALE)

    if img_opt is None or img_sem is None:
        raise RuntimeError("Images not found")

    img_sem = cv2.cvtColor(img_sem, cv2.COLOR_GRAY2BGR)

    cv2.namedWindow("optical")
    cv2.namedWindow("sem")
    cv2.namedWindow("overlay")

    cv2.setMouseCallback("optical", mouse_opt)
    cv2.setMouseCallback("sem", mouse_sem)

    print("""
Controls
--------
Left click     : add point
Drag point     : move point
D              : delete last pair
C              : clear all
S              : save overlay
ESC            : quit
""")

    while True:
        vis_opt = draw_points(img_opt, cp.opt)
        vis_sem = draw_points(img_sem, cp.sem)

        overlay = compute_overlay(img_opt, img_sem)

        cv2.imshow("optical", vis_opt)
        cv2.imshow("sem", vis_sem)
        cv2.imshow("overlay", overlay)

        key = cv2.waitKey(20) & 0xFF

        if key == 27:
            break

        elif key == ord("d"):
            if cp.opt:
                cp.opt.pop()
            if cp.sem:
                cp.sem.pop()

        elif key == ord("c"):
            cp.opt.clear()
            cp.sem.clear()

        elif key == ord("s"):
            cv2.imwrite("overlay.png", overlay)
            print("Saved overlay.png")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    IMG_SEM = Path("/home/m/Dropbox/ZUMI/zakazky/2611_Trinity/sem_result/2611v4.jpg")
    IMG_OPT = Path(
        "/home/m/Dropbox/ZUMI/zakazky/2611_Trinity/sem_result/2611v4_uv_s_N10_2604091520470161_25.jpg"
    )
    main()
