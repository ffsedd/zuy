from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

# =====================================================
# CONFIG
# =====================================================

OPT_PATH = Path(
    "/home/m/Dropbox/ZUMI/zakazky/2611_Trinity/sem_result/2611v4_uv_s_N10_2604091520470161_.jpg"
)
SEM_PATH = Path("/home/m/Dropbox/ZUMI/zakazky/2611_Trinity/sem_result/2611v4.jpg")

OPT_DISPLAY_SCALE = 0.25
SEM_FINAL_SCALE = 4.0
ALPHA = 0.5

Point = Tuple[int, int]


# =====================================================
# DATA
# =====================================================


@dataclass
class CP:
    opt: List[Point]
    sem: List[Point]

    def valid(self):
        return len(self.opt) >= 4 and len(self.opt) == len(self.sem)


cp = CP([], [])
drag_idx: Optional[int] = None
drag_mode: Optional[str] = None


# =====================================================
# UTIL
# =====================================================


def resize(img, scale):
    return cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


def draw_pts(img, pts):
    out = img.copy()
    for i, (x, y) in enumerate(pts):
        cv2.circle(out, (x, y), 6, (0, 0, 255), 2)
        cv2.putText(out, str(i + 1), (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out


def nearest(pt, pts):
    if not pts:
        return None
    d = [np.hypot(px - pt[0], py - pt[1]) for px, py in pts]
    i = int(np.argmin(d))
    if d[i] < 15:
        return i
    return None


# =====================================================
# HOMOGRAPHY
# =====================================================


def homography_disp():
    if not cp.valid():
        return None

    src = np.array(cp.sem, np.float32)
    dst = np.array(cp.opt, np.float32)

    H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
    return H


def preview_overlay(opt_disp, sem_disp):
    H = homography_disp()
    if H is None:
        return opt_disp

    warped = cv2.warpPerspective(
        sem_disp,
        H,
        (opt_disp.shape[1], opt_disp.shape[0]),
    )

    return cv2.addWeighted(opt_disp, 1 - ALPHA, warped, ALPHA, 0)


# =====================================================
# MOUSE
# =====================================================


def mouse_opt(event, x, y, *_):
    global drag_idx, drag_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        idx = nearest((x, y), cp.opt)
        if idx is not None:
            drag_idx = idx
            drag_mode = "opt"
        else:
            cp.opt.append((x, y))

    elif event == cv2.EVENT_MOUSEMOVE and drag_mode == "opt":
        cp.opt[drag_idx] = (x, y)  # type: ignore

    elif event == cv2.EVENT_LBUTTONUP:
        drag_idx = None
        drag_mode = None


def mouse_sem(event, x, y, *_):
    global drag_idx, drag_mode

    if event == cv2.EVENT_LBUTTONDOWN:
        idx = nearest((x, y), cp.sem)
        if idx is not None:
            drag_idx = idx
            drag_mode = "sem"
        else:
            cp.sem.append((x, y))

    elif event == cv2.EVENT_MOUSEMOVE and drag_mode == "sem":
        cp.sem[drag_idx] = (x, y)  # type: ignore

    elif event == cv2.EVENT_LBUTTONUP:
        drag_idx = None
        drag_mode = None


def auto_init_points(opt_full, sem_full, max_points=8):
    """
    Suggest initial correspondences using ORB + homography.
    Returns OPT points and SEM points in DISPLAY coordinates.
    """

    orb = cv2.ORB_create(5000)  # type: ignore

    opt_gray = cv2.cvtColor(opt_full, cv2.COLOR_BGR2GRAY)
    sem_gray = cv2.cvtColor(sem_full, cv2.COLOR_BGR2GRAY)

    kp1, des1 = orb.detectAndCompute(opt_gray, None)
    kp2, des2 = orb.detectAndCompute(sem_gray, None)

    if des1 is None or des2 is None:
        print("Auto-init failed: no descriptors")
        return [], []

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des2, des1)  # SEM → OPT

    matches = sorted(matches, key=lambda x: x.distance)

    if len(matches) < 8:
        print("Not enough matches for auto-init")
        return [], []

    # take best matches but spatially diverse
    pts_sem = []
    pts_opt = []

    used_opt = set()

    for m in matches:
        p_sem = kp2[m.queryIdx].pt
        p_opt = kp1[m.trainIdx].pt

        if len(pts_sem) >= max_points:
            break

        # avoid duplicates (simple spatial gating)
        if any(np.hypot(p_opt[0] - u[0], p_opt[1] - u[1]) < 30 for u in used_opt):
            continue

        used_opt.add(p_opt)

        pts_sem.append((int(p_sem[0]), int(p_sem[1])))
        pts_opt.append((int(p_opt[0]), int(p_opt[1])))

    print(f"[AUTO] initialized {len(pts_opt)} points")

    return pts_opt, pts_sem


# =====================================================
# FINAL EXPORT
# =====================================================


def export_full(opt_full, sem_full, warp_out_path, blend_out_path):

    H_disp = homography_disp()
    if H_disp is None:
        print("Need ≥4 pairs.")
        return

    # scale from full OPT → display OPT
    s = OPT_DISPLAY_SCALE

    # S_opt = np.array(
    #     [
    #         [s, 0, 0],
    #         [0, s, 0],
    #         [0, 0, 1],
    #     ],
    #     dtype=np.float64,
    # )
    # print(f"S_opt:\n{S_opt}")

    S_opt_inv = np.array(
        [
            [1 / s, 0, 0],
            [0, 1 / s, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )

    # =====================================================
    # CRITICAL FIX:
    # H_disp maps SEM_full → OPT_disp
    # so full transform is:
    # SEM_full → OPT_full
    # =====================================================

    H_full = S_opt_inv @ H_disp

    # warp OPT into SEM space (invert direction)
    H_opt_to_sem = np.linalg.inv(H_full)

    sem_h, sem_w = sem_full.shape[:2]

    warped_opt = cv2.warpPerspective(
        opt_full,
        H_opt_to_sem,
        (sem_w, sem_h),
    )
    cv2.imwrite(blend_out_path.parent / f"{blend_out_path.stem}_{0.0}.jpg", warped_opt)

    for alpha in [0.8, 0.5, 0.3]:
        overlay = cv2.addWeighted(
            warped_opt,
            1 - alpha,
            sem_full,
            ALPHA,
            0,
        )

        cv2.imwrite(blend_out_path.parent / f"{blend_out_path.stem}_{alpha}.jpg", overlay)

    print("Saved:")
    print(f"{warp_out_path}")
    print(f"{blend_out_path}")


# =====================================================
# MAIN
# =====================================================


def semvis(sem_path: Path = SEM_PATH, opt_path: Path = OPT_PATH):

    opt_full = cv2.imread(str(opt_path))
    sem_full = cv2.imread(str(sem_path))

    if opt_full is None or sem_full is None:
        raise RuntimeError("Image load failed")

    opt_disp = resize(opt_full, OPT_DISPLAY_SCALE)
    sem_disp = sem_full.copy()

    # =====================================================
    # AUTO INITIALIZATION (NEW)
    # =====================================================

    init_opt, init_sem = auto_init_points(opt_full, sem_full)

    # scale OPT points to display space
    cp.opt = [(int(x * OPT_DISPLAY_SCALE), int(y * OPT_DISPLAY_SCALE)) for x, y in init_opt]
    cp.sem = init_sem

    cv2.namedWindow("OPT")
    cv2.namedWindow("SEM")
    cv2.namedWindow("Overlay")

    cv2.setMouseCallback("OPT", mouse_opt)
    cv2.setMouseCallback("SEM", mouse_sem)

    print("""
Controls
--------
Click       add point
Drag        move point
D           delete last pair
C           clear
E           export full-resolution result
ESC         quit
""")

    while True:
        cv2.imshow("OPT", draw_pts(opt_disp, cp.opt))
        cv2.imshow("SEM", draw_pts(sem_disp, cp.sem))
        cv2.imshow("Overlay", preview_overlay(opt_disp, sem_disp))

        k = cv2.waitKey(20) & 0xFF

        if k == 27:
            break
        elif k == ord("d"):
            if cp.opt:
                cp.opt.pop()
            if cp.sem:
                cp.sem.pop()
        elif k == ord("c"):
            cp.opt.clear()
            cp.sem.clear()
        elif k == ord("e"):
            export_full(
                opt_full,
                sem_full,
                opt_path.parent / f"{opt_path.stem}_warped.jpg",
                sem_path.parent / f"{sem_path.stem}_blend.jpg",
            )
            cv2.destroyAllWindows()


def parsearg():
    p = argparse.ArgumentParser(description="SEM ↔ Optical image registration tool")
    p.add_argument("sempath", type=Path, help="SEM image path")
    p.add_argument("vispath", type=Path, help="VIS image path")

    p.add_argument("--scale", type=float, default=OPT_DISPLAY_SCALE)
    p.add_argument("--alpha", type=float, default=ALPHA)

    p.add_argument("--out", type=Path, default=Path("."))
    p.add_argument("--no-gui", action="store_true")
    p.add_argument("--export", action="store_true")

    return p


def main():
    global OPT_DISPLAY_SCALE, ALPHA

    args = parsearg().parse_args()

    OPT_DISPLAY_SCALE = args.scale
    ALPHA = args.alpha

    opt_full = cv2.imread(str(args.opt))
    sem_full = cv2.imread(str(args.sem))

    if opt_full is None or sem_full is None:
        raise RuntimeError("Image load failed")

    if args.no_gui:
        # headless mode requires predefined points (future extension)
        raise NotImplementedError("Headless mode needs stored landmarks (add YAML support next).")

    semvis(args.sempath, args.vispath)


if __name__ == "__main__":
    main()
