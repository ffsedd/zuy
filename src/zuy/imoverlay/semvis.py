from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

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
# TODO(refactor): confirm intent for SEM_FINAL_SCALE — currently defined but unused anywhere
SEM_FINAL_SCALE = 4.0
ALPHA = 0.5

Point = Tuple[int, int]


# =====================================================
# CONSTANTS
# =====================================================

DRAG_HIT_RADIUS_PX = 15
ORB_SPATIAL_DEDUP_PX = 30
ORB_FEATURE_COUNT = 5000
AUTO_INIT_MAX_POINTS = 8
POINT_MARKER_RADIUS_PX = 6
POINT_LABEL_FONT_SCALE = 0.6
EXPORT_ALPHA_LEVELS = (0.8, 0.5, 0.3)

# Boundary-based auto-init tuning
BOUNDARY_BLUR_KSIZE: int = 5          # GaussianBlur kernel (must be odd)
BOUNDARY_EPSILON_FRAC: float = 0.02   # polygon approx fraction of arc-length
BOUNDARY_MIN_CORNERS: int = 3         # minimum usable corners per image
BOUNDARY_MAX_CORNERS: int = AUTO_INIT_MAX_POINTS  # cap (reuses existing constant)


# =====================================================
# DATA
# =====================================================


@dataclass
class RegistrationState:
    opt: List[Point] = field(default_factory=list)
    sem: List[Point] = field(default_factory=list)
    drag_idx: Optional[int] = None
    drag_mode: Optional[str] = None

    def valid(self) -> bool:
        return len(self.opt) >= 4 and len(self.opt) == len(self.sem)


# =====================================================
# UTIL
# =====================================================


def resize(img: np.ndarray, scale: float) -> np.ndarray:
    return cv2.resize(
        img,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA,
    )


def draw_pts(img: np.ndarray, pts: List[Point]) -> np.ndarray:
    out = img.copy()
    for i, (x, y) in enumerate(pts):
        cv2.circle(out, (x, y), POINT_MARKER_RADIUS_PX, (0, 0, 255), 2)
        cv2.putText(out, str(i + 1), (x + 5, y + 5), cv2.FONT_HERSHEY_SIMPLEX, POINT_LABEL_FONT_SCALE, (0, 255, 0), 2)
    return out


def nearest(pt: Point, pts: List[Point]) -> Optional[int]:
    if not pts:
        return None
    d = [np.hypot(px - pt[0], py - pt[1]) for px, py in pts]
    i = int(np.argmin(d))
    if d[i] < DRAG_HIT_RADIUS_PX:
        return i
    return None


# =====================================================
# HOMOGRAPHY
# =====================================================


def homography_disp(state: RegistrationState) -> Optional[np.ndarray]:
    if not state.valid():
        return None

    src = np.array(state.sem, np.float32)
    dst = np.array(state.opt, np.float32)

    H, _ = cv2.findHomography(src, dst, cv2.RANSAC)
    return H


def preview_overlay(
    state: RegistrationState,
    opt_disp: np.ndarray,
    sem_disp: np.ndarray,
    alpha: float,
) -> np.ndarray:
    H = homography_disp(state)
    if H is None:
        return opt_disp

    warped = cv2.warpPerspective(
        sem_disp,
        H,
        (opt_disp.shape[1], opt_disp.shape[0]),
    )

    return cv2.addWeighted(opt_disp, 1 - alpha, warped, alpha, 0)


# =====================================================
# MOUSE
# =====================================================


def make_mouse_handler(state: RegistrationState, which: str):
    target_pts = state.opt if which == "opt" else state.sem

    def handler(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            idx = nearest((x, y), target_pts)
            if idx is not None:
                state.drag_idx = idx
                state.drag_mode = which
            else:
                target_pts.append((x, y))

        elif event == cv2.EVENT_MOUSEMOVE and state.drag_mode == which and state.drag_idx is not None:
            target_pts[state.drag_idx] = (x, y) # type: ignore

        elif event == cv2.EVENT_LBUTTONUP:
            state.drag_idx = None
            state.drag_mode = None
    return handler


def _segment_sample_mask(img: np.ndarray, *, blur_ksize: int = BOUNDARY_BLUR_KSIZE) -> np.ndarray:
    """
    Return a binary mask of the sample region (largest connected component
    after Otsu threshold).  Works on grayscale or BGR input.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    gray = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask
    largest = max(contours, key=cv2.contourArea)
    out = np.zeros_like(mask)
    cv2.drawContours(out, [largest], -1, 255, thickness=cv2.FILLED)
    return out


def _boundary_corners(
    mask: np.ndarray,
    *,
    epsilon_frac: float = BOUNDARY_EPSILON_FRAC,
    max_corners: int = BOUNDARY_MAX_CORNERS,
) -> List[Point]:
    """
    Polygon-approximate the outer contour of *mask* and return corner points
    sorted by angle around the centroid (clockwise from the topmost point).
    Returns at most *max_corners* points.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    largest = max(contours, key=cv2.contourArea)
    epsilon = epsilon_frac * cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, epsilon, True)
    pts = [(int(p[0][0]), int(p[0][1])) for p in approx]

    # Sort by angle around centroid for consistent ordering across modalities
    cx = sum(x for x, _ in pts) / len(pts)
    cy = sum(y for _, y in pts) / len(pts)
    pts.sort(key=lambda p: np.arctan2(p[1] - cy, p[0] - cx))

    return pts[:max_corners]


def auto_init_points(
    opt_full: np.ndarray,
    sem_full: np.ndarray,
    max_points: int = AUTO_INIT_MAX_POINTS,
) -> Tuple[List[Point], List[Point]]:
    """
    Suggest initial correspondences using ORB + homography.
    Returns OPT points and SEM points in DISPLAY coordinates.
    """

    orb = cv2.ORB_create(ORB_FEATURE_COUNT)  # type: ignore

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

    if len(matches) < max_points:
        print("Not enough matches for auto-init")
        return [], []

    # take best matches but spatially diverse
    pts_sem = []
    pts_opt = []

    used_opt: List[Point] = []

    for m in matches:
        p_sem = kp2[m.queryIdx].pt
        p_opt = kp1[m.trainIdx].pt

        if len(pts_sem) >= max_points:
            break

        # avoid duplicates (simple spatial gating)
        if any(np.hypot(p_opt[0] - u[0], p_opt[1] - u[1]) < ORB_SPATIAL_DEDUP_PX for u in used_opt):
            continue

        used_opt.append(p_opt)

        pts_sem.append((int(p_sem[0]), int(p_sem[1])))
        pts_opt.append((int(p_opt[0]), int(p_opt[1])))

    print(f"[AUTO] initialized {len(pts_opt)} points")

    return pts_opt, pts_sem


def auto_init_points_boundary(
    opt_full: np.ndarray,
    sem_full: np.ndarray,
    max_points: int = BOUNDARY_MAX_CORNERS,
) -> Tuple[List[Point], List[Point]]:
    """
    Suggest initial correspondences using sample-boundary corners.

    Segments the sample outline independently in each image via Otsu threshold,
    polygon-approximates the largest contour to extract corners, then matches
    corners by angular order around the centroid.

    Returns OPT points in **full-resolution** coordinates and SEM points in
    **native** (full-resolution) coordinates — same contract as auto_init_points().

    Limitations
    -----------
    - Requires the sample to be distinctly brighter or darker than the background
      in both images (Otsu assumption).
    - Corner ordering is purely angular; a 90° rotation ambiguity is possible if
      the sample is near-square.  Resolve with one manual correction after init.
    - Fails gracefully (returns [], []) if either image has fewer than
      BOUNDARY_MIN_CORNERS usable corners.
    """
    mask_opt = _segment_sample_mask(opt_full)
    mask_sem = _segment_sample_mask(sem_full)

    corners_opt = _boundary_corners(mask_opt, max_corners=max_points)
    corners_sem = _boundary_corners(mask_sem, max_corners=max_points)

    if len(corners_opt) < BOUNDARY_MIN_CORNERS or len(corners_sem) < BOUNDARY_MIN_CORNERS:
        print(
            f"[BOUNDARY] too few corners — OPT:{len(corners_opt)} SEM:{len(corners_sem)} "
            f"(need ≥{BOUNDARY_MIN_CORNERS})"
        )
        return [], []

    # Pair corners by angular rank; use the shorter list as the limit
    n = min(len(corners_opt), len(corners_sem), max_points)
    pts_opt = corners_opt[:n]
    pts_sem = corners_sem[:n]

    print(f"[BOUNDARY] initialized {n} point pairs from boundary corners")
    return pts_opt, pts_sem


# =====================================================
# FINAL EXPORT
# =====================================================


def compute_full_homography(H_disp: np.ndarray, display_scale: float) -> np.ndarray:
    """
    Computes the full-resolution homography (SEM_full -> OPT_full)
    from a display-scaled homography (SEM_full -> OPT_disp).
    """
    s_inv = 1 / display_scale
    S_opt_inv = np.array(
        [
            [s_inv, 0, 0],
            [0, s_inv, 0],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    # H_disp maps SEM_full -> OPT_disp
    # so full transform is: SEM_full -> OPT_full
    H_full = S_opt_inv @ H_disp
    return H_full


def warp_optical_to_sem(opt_full: np.ndarray, sem_full_shape: Tuple[int, int], H_full: np.ndarray) -> np.ndarray:
    """
    Warps the full optical image into the SEM image space using the full homography.
    """
    # Invert H_full to warp OPT_full -> SEM_full space
    H_opt_to_sem = np.linalg.inv(H_full)
    sem_h, sem_w = sem_full_shape
    warped_opt = cv2.warpPerspective(
        opt_full,
        H_opt_to_sem,
        (sem_w, sem_h),
    )
    return warped_opt


def save_overlays(
    warped_opt: np.ndarray,
    sem_full: np.ndarray,
    output_dir: Path,
    output_stem: str,
    blend_alphas: Tuple[float, ...],
    overlay_alpha: float,
) -> List[Path]:
    """
    Saves the warped optical image and blended overlays with SEM image.
    The warped image is saved with a '_0.0.jpg' suffix, and blended images
    use the provided blend_alphas.
    """
    saved_paths: List[Path] = []

    # Save the warped optical image (alpha 0.0 equivalent)
    warped_output_path = output_dir / f"{output_stem}_{0.0}.jpg"
    cv2.imwrite(str(warped_output_path), warped_opt)
    saved_paths.append(warped_output_path)

    # Save blended images
    for alpha in blend_alphas:
        overlay = cv2.addWeighted(
            warped_opt,
            1 - alpha,
            sem_full,
            overlay_alpha,
            0,
        )
        blended_output_path = output_dir / f"{output_stem}_{alpha}.jpg"
        cv2.imwrite(str(blended_output_path), overlay)
        saved_paths.append(blended_output_path)

    return saved_paths


def export_full(
    state: RegistrationState,
    opt_full: np.ndarray,
    sem_full: np.ndarray,
    warp_out_path: Path,
    blend_out_path: Path,
    display_scale: float,
    alpha: float,
):

    H_disp = homography_disp(state)
    if H_disp is None:
        print("Need ≥4 pairs.")
        return

    H_full = compute_full_homography(H_disp, display_scale)
    warped_opt = warp_optical_to_sem(opt_full, sem_full.shape[:2], H_full)

    # The original code's printing implies these paths, but the saving logic
    # uses blend_out_path's parent and stem for all output files.
    # The warp_out_path itself isn't used as an output filename.
    # We will print the paths as per the original print statements.

    saved_files = save_overlays(
        warped_opt,
        sem_full,
        blend_out_path.parent,
        blend_out_path.stem,
        EXPORT_ALPHA_LEVELS,
        alpha,
    )

    print("Saved:")
    print(f"{warp_out_path}") # This is still printed, though not used for the actual saved filename
    for p in saved_files:
        print(f"{p}")


# =====================================================
# MAIN
# =====================================================


def semvis(sem_path: Path, opt_path: Path, *, display_scale: float, alpha: float) -> None:

    opt_full = cv2.imread(str(opt_path))
    sem_full = cv2.imread(str(sem_path))

    if opt_full is None or sem_full is None:
        raise RuntimeError("Image load failed")

    opt_disp = resize(opt_full, display_scale)
    # TODO(refactor): confirm SEM display is intentionally unscaled
    # optical image is scaled by display_scale, but SEM image is displayed at native resolution
    sem_disp = sem_full.copy()

    state = RegistrationState()

    # =====================================================
    # AUTO INITIALIZATION (NEW)
    # =====================================================

    def _apply_init(pts_opt_full: List[Point], pts_sem_native: List[Point]) -> None:
        state.opt = [(int(x * display_scale), int(y * display_scale)) for x, y in pts_opt_full]
        state.sem = list(pts_sem_native)

    # Startup: attempt boundary-based init; fall back to ORB
    _init_opt, _init_sem = auto_init_points_boundary(opt_full, sem_full)
    if not _init_opt:
        _init_opt, _init_sem = auto_init_points(opt_full, sem_full)
    _apply_init(_init_opt, _init_sem)

    cv2.namedWindow("VIS")
    cv2.namedWindow("SEM")
    cv2.namedWindow("Overlay")

    cv2.setMouseCallback("VIS", make_mouse_handler(state, "opt"))
    cv2.setMouseCallback("SEM", make_mouse_handler(state, "sem"))

    print("""
Controls
--------
Click       add point
Drag        move point
D           delete last pair
C           clear all points
A           auto-init: ORB features (original)
B           auto-init: boundary corners (new)
E           export full-resolution result
ESC         quit
""")

    while True:
        cv2.imshow("VIS", draw_pts(opt_disp, state.opt))
        cv2.imshow("SEM", draw_pts(sem_disp, state.sem))
        cv2.imshow("Overlay", preview_overlay(state, opt_disp, sem_disp, alpha))

        k = cv2.waitKey(20) & 0xFF

        if k == 27:
            break
        elif k == ord("d"):
            if state.opt:
                state.opt.pop()
            if state.sem:
                state.sem.pop()
        elif k == ord("c"):
            state.opt.clear()
            state.sem.clear()
        elif k == ord("a"):
            _pts_opt, _pts_sem = auto_init_points(opt_full, sem_full)
            _apply_init(_pts_opt, _pts_sem)
        elif k == ord("b"):
            _pts_opt, _pts_sem = auto_init_points_boundary(opt_full, sem_full)
            _apply_init(_pts_opt, _pts_sem)
        elif k == ord("e"):
            export_full(
                state,
                opt_full,
                sem_full,
                opt_path.parent / f"{opt_path.stem}_warped.jpg",
                sem_path.parent / f"{sem_path.stem}_blend.jpg",
                display_scale,
                alpha,
            )
            cv2.destroyAllWindows()


def parsearg():
    # TODO(refactor): confirm intent for --out/--export/--no-gui
    # --out is parsed but never used.
    # --export is parsed but never used (only the in-GUI `e` key triggers export).
    # --no-gui raises `NotImplementedError` referencing a future headless
    # mode that needs stored landmark YAML support, which doesn't exist.
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
    args = parsearg().parse_args()

    # TODO(refactor): confirm intent for --out/--export/--no-gui
    # --out is parsed but never used.
    # --export is parsed but never used (only the in-GUI `e` key triggers export).
    # --no-gui raises `NotImplementedError` referencing a future headless
    # mode that needs stored landmark YAML support, which doesn't exist.

    if args.no_gui:
        # headless mode requires predefined points (future extension)
        raise NotImplementedError("Headless mode needs stored landmarks (add YAML support next).")

    semvis(args.sempath, args.vispath, display_scale=args.scale, alpha=args.alpha)


if __name__ == "__main__":
    main()
