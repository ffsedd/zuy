#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageTk

# ============================================================================
# Logging
# ============================================================================

LOGGER = logging.getLogger("semvis")


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ============================================================================
# Configuration
# ============================================================================

DEFAULT_VIS_PATH = Path(
    "/home/m/Dropbox/ZUMI/zakazky/2611_Trinity/sem_result/2611v4_uv_s_N10_2604091520470161_.jpg"
)

DEFAULT_SEM_PATH = Path("/home/m/Dropbox/ZUMI/zakazky/2611_Trinity/sem_result/2611v4.jpg")

VIS_DISPLAY_SCALE = 0.25
ALPHA = 0.5

DRAG_HIT_RADIUS_PX = 15

ORB_SPATIAL_DEDUP_PX = 30
ORB_FEATURE_COUNT = 5000
AUTO_INIT_MAX_POINTS = 4

MARKER_RADIUS = 7
MARKER_THICKNESS = 2

LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.6
LABEL_THICKNESS = 2

EXPORT_ALPHA_LEVELS = (0.3,)

BOUNDARY_THRESHOLD = 40
BOUNDARY_MIN_AREA_FRACTION = 0.01
BOUNDARY_MAX_CORNERS = 8


Point = tuple[int, int]


# ============================================================================
# Registration state
# ============================================================================


@dataclass(frozen=True)
class RegistrationPoints:
    """
    Canonical registration coordinates.

    Both point sets are stored in full-resolution image coordinates.

    vis[i] corresponds to sem[i].
    """

    vis: tuple[Point, ...]
    sem: tuple[Point, ...]
    version: int = 2

    def valid(self) -> bool:
        return (
            len(self.vis) >= 4
            and len(self.vis) == len(self.sem)
            and all(x >= 0 and y >= 0 for x, y in self.vis)
            and all(x >= 0 and y >= 0 for x, y in self.sem)
        )


@dataclass
class RegistrationState:
    """
    Mutable GUI state.

    Coordinates remain in full-resolution image coordinates.

    Points are paired by list index:

        vis[0] <-> sem[0]
        vis[1] <-> sem[1]
        ...

    The two lists are intentionally allowed to have different lengths while
    the user is creating a registration.
    """

    vis: list[Point]
    sem: list[Point]
    drag_idx: int | None = None
    drag_mode: str | None = None

    def valid(self) -> bool:
        return (
            len(self.vis) >= 4
            and len(self.vis) == len(self.sem)
            and all(x >= 0 and y >= 0 for x, y in self.vis)
            and all(x >= 0 and y >= 0 for x, y in self.sem)
        )

    def pair_count(self) -> int:
        return min(len(self.vis), len(self.sem))

    def registration(self) -> RegistrationPoints:
        return RegistrationPoints(
            vis=tuple(self.vis),
            sem=tuple(self.sem),
        )

    def set_registration(
        self,
        registration: RegistrationPoints,
    ) -> None:
        self.vis = list(registration.vis)
        self.sem = list(registration.sem)
        self.drag_idx = None
        self.drag_mode = None


# ============================================================================
# Paths / JSON persistence
# ============================================================================


def registration_path(
    sem_path: Path,
    vis_path: Path,
) -> Path:
    path = sem_path.parent / (f"{sem_path.stem}_{vis_path.stem}.registration.json")

    LOGGER.debug("Registration path: %s", path)

    return path


def save_registration(
    path: Path,
    registration: RegistrationPoints,
) -> None:
    if not registration.valid():
        raise ValueError("Cannot save an invalid registration.")

    LOGGER.info(
        "Saving registration: %s (%d point pairs)",
        path,
        len(registration.vis),
    )

    data = {
        "version": registration.version,
        "vis": [[x, y] for x, y in registration.vis],
        "sem": [[x, y] for x, y in registration.sem],
    }

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp",
    )

    temporary.write_text(
        json.dumps(data, indent=2) + "\n",
        encoding="utf-8",
    )

    temporary.replace(path)

    LOGGER.debug(
        "Registration written atomically via %s",
        temporary,
    )


def _parse_points(
    value: object,
    name: str,
) -> tuple[Point, ...]:
    if not isinstance(value, list):
        raise ValueError(
            f"JSON field {name!r} must be a list.",
        )

    points: list[Point] = []

    for index, item in enumerate(value):
        if (
            not isinstance(item, list)
            or len(item) != 2
            or not all(isinstance(v, (int, float)) for v in item)
        ):
            raise ValueError(
                f"Invalid {name}[{index}]: expected [x, y].",
            )

        x = int(round(float(item[0])))
        y = int(round(float(item[1])))

        if x < 0 or y < 0:
            raise ValueError(
                f"Invalid {name}[{index}]: coordinates must be >= 0.",
            )

        points.append((x, y))

    return tuple(points)


def load_registration(
    path: Path,
) -> RegistrationPoints:
    LOGGER.info(
        "Loading registration: %s",
        path,
    )

    data = json.loads(
        path.read_text(
            encoding="utf-8",
        ),
    )

    if not isinstance(data, dict):
        raise ValueError(
            "Registration JSON root must be an object.",
        )

    version = data.get("version")

    if version != 2:
        raise ValueError(
            f"Unsupported registration JSON version: {version!r}. Expected version 2.",
        )

    vis = _parse_points(
        data.get("vis"),
        "vis",
    )

    sem = _parse_points(
        data.get("sem"),
        "sem",
    )

    registration = RegistrationPoints(
        vis=vis,
        sem=sem,
        version=version,
    )

    LOGGER.info(
        "Loaded %d VIS / %d SEM points",
        len(vis),
        len(sem),
    )

    if len(vis) != len(sem):
        raise ValueError(
            f"Registration contains {len(vis)} VIS points and {len(sem)} SEM points.",
        )

    if not registration.valid():
        LOGGER.warning(
            "Loaded registration is not usable: need at least 4 matching point pairs.",
        )

    LOGGER.debug(
        "VIS points: %s",
        vis,
    )

    LOGGER.debug(
        "SEM points: %s",
        sem,
    )

    return registration


# ============================================================================
# Image helpers
# ============================================================================


def load_image(path: Path) -> np.ndarray:
    LOGGER.info(
        "Loading image: %s",
        path,
    )

    image = cv2.imread(
        str(path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"Cannot read image: {path}",
        )

    LOGGER.info(
        "Loaded %s: width=%d height=%d channels=%d",
        path.name,
        image.shape[1],
        image.shape[0],
        image.shape[2] if image.ndim == 3 else 1,
    )

    LOGGER.debug(
        "%s dtype=%s shape=%s",
        path.name,
        image.dtype,
        image.shape,
    )

    return image


def resize_for_display(
    image: np.ndarray,
    scale: float,
) -> np.ndarray:
    if scale <= 0:
        raise ValueError(
            "Display scale must be > 0.",
        )

    if scale == 1:
        return image.copy()

    width = max(
        1,
        round(image.shape[1] * scale),
    )

    height = max(
        1,
        round(image.shape[0] * scale),
    )

    return cv2.resize(
        image,
        (width, height),
        interpolation=(cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR),
    )


def full_to_display_point(
    point: Point,
    scale: float,
) -> Point:
    return (
        round(point[0] * scale),
        round(point[1] * scale),
    )


def display_to_full_point(
    point: Point,
    scale: float,
) -> Point:
    return (
        round(point[0] / scale),
        round(point[1] / scale),
    )





# ============================================================================
# Point drawing / interaction
# ============================================================================


def draw_points(
    image: np.ndarray,
    points: list[Point] | tuple[Point, ...],
    scale: float = 1.0,
) -> np.ndarray:
    output = image.copy()

    for index, point in enumerate(points):
        x, y = full_to_display_point(
            point,
            scale,
        )

        cv2.circle(
            output,
            (x, y),
            MARKER_RADIUS,
            (0, 255, 0),
            MARKER_THICKNESS,
        )

        cv2.putText(
            output,
            str(index + 1),
            (x + 10, y - 10),
            LABEL_FONT,
            LABEL_SCALE,
            (0, 255, 0),
            LABEL_THICKNESS,
            cv2.LINE_AA,
        )

    return output


def canvas_to_full_point(
    canvas_point: tuple[int | float, int | float],
    scale: float,
    offset: tuple[float, float] = (0.0, 0.0),
) -> Point:
    return (
        round((canvas_point[0] - offset[0]) / scale),
        round((canvas_point[1] - offset[1]) / scale),
    )


def nearest_point_index(
    points: list[Point],
    display_point: Point,
    display_scale: float,
    offset: tuple[float, float] = (0.0, 0.0),
    radius: float = DRAG_HIT_RADIUS_PX,
) -> int | None:
    if not points:
        return None

    best_index: int | None = None
    best_distance_sq = radius * radius

    x, y = display_point
    ox, oy = offset

    for index, point in enumerate(points):
        px, py = full_to_display_point(
            point,
            display_scale,
        )
        px += ox
        py += oy

        dx = px - x
        dy = py - y

        distance_sq = dx * dx + dy * dy

        if distance_sq <= best_distance_sq:
            best_distance_sq = distance_sq
            best_index = index

    return best_index


# ============================================================================
# Homography
# ============================================================================


def homography_full(
    state: RegistrationState,
) -> np.ndarray | None:
    """
    Calculate VIS(full) -> SEM(full).

    SEM is the fixed/reference coordinate system.
    """

    if not state.valid():
        LOGGER.debug(
            "Homography unavailable: invalid registration (VIS=%d SEM=%d)",
            len(state.vis),
            len(state.sem),
        )

        return None

    src = np.asarray(
        state.vis,
        dtype=np.float32,
    )

    dst = np.asarray(
        state.sem,
        dtype=np.float32,
    )

    homography, mask = cv2.findHomography(
        src,
        dst,
        method=cv2.RANSAC,
        ransacReprojThreshold=5.0,
    )

    if homography is None:
        LOGGER.warning(
            "cv2.findHomography returned None.",
        )
        return None

    if mask is not None:
        inliers = int(mask.sum())
        total = len(mask)

        LOGGER.info(
            "Homography RANSAC: %d/%d inliers",
            inliers,
            total,
        )

        if inliers < 4:
            LOGGER.warning(
                "Homography rejected: fewer than 4 RANSAC inliers.",
            )
            return None

    LOGGER.debug(
        "VIS -> SEM homography:\n%s",
        homography,
    )

    return homography


def homography_display(
    state: RegistrationState,
    vis_scale: float,
    sem_scale: float = 1.0,
) -> np.ndarray | None:
    """
    Convert VIS(full) -> SEM(full) into:

        VIS(display) -> SEM(display)
    """

    homography = homography_full(
        state,
    )

    if homography is None:
        return None

    scale_vis_inv = np.array(
        [
            [1.0 / vis_scale, 0.0, 0.0],
            [0.0, 1.0 / vis_scale, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    if sem_scale == 1.0:
        return homography @ scale_vis_inv

    scale_sem_mat = np.array(
        [
            [sem_scale, 0.0, 0.0],
            [0.0, sem_scale, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return scale_sem_mat @ homography @ scale_vis_inv


# ============================================================================
# Preview / export
# ============================================================================


def preview_overlay(
    vis_display: np.ndarray,
    sem_display: np.ndarray,
    state: RegistrationState,
    vis_scale: float,
    alpha: float,
    sem_scale: float = 1.0,
) -> np.ndarray:
    homography = homography_display(
        state,
        vis_scale,
        sem_scale,
    )

    if homography is None:
        return sem_display.copy()

    sem_height, sem_width = sem_display.shape[:2]

    warped_vis = cv2.warpPerspective(
        vis_display,
        homography,
        (sem_width, sem_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    return cv2.addWeighted(
        sem_display,
        1.0 - alpha,
        warped_vis,
        alpha,
        0.0,
    )


def export_full(
    vis_full: np.ndarray,
    sem_full: np.ndarray,
    state: RegistrationState,
    warp_out_path: Path,
    blend_out_path: Path,
    overlay_alpha: float,
    blend_alphas: tuple[float, ...] = EXPORT_ALPHA_LEVELS,
) -> None:
    LOGGER.info(
        "Starting full-resolution export.",
    )

    homography = homography_full(
        state,
    )

    if homography is None:
        raise RuntimeError(
            "Could not calculate VIS -> SEM homography.",
        )

    sem_height, sem_width = sem_full.shape[:2]

    warped_vis = cv2.warpPerspective(
        vis_full,
        homography,
        (sem_width, sem_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    warp_out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    blend_out_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    LOGGER.info(
        "Writing warped VIS: %s",
        warp_out_path,
    )

    if not cv2.imwrite(
        str(warp_out_path),
        warped_vis,
    ):
        raise RuntimeError(
            f"Could not write warped VIS image: {warp_out_path}",
        )

    blended = cv2.addWeighted(
        sem_full,
        1.0 - overlay_alpha,
        warped_vis,
        overlay_alpha,
        0.0,
    )

    LOGGER.info(
        "Writing overlay: %s (alpha=%.2f)",
        blend_out_path,
        overlay_alpha,
    )

    if not cv2.imwrite(
        str(blend_out_path),
        blended,
    ):
        raise RuntimeError(
            f"Could not write overlay image: {blend_out_path}",
        )

    for alpha in blend_alphas:
        output = cv2.addWeighted(
            sem_full,
            1.0 - alpha,
            warped_vis,
            alpha,
            0.0,
        )

        alpha_name = f"{alpha:.1f}".replace(
            ".",
            "",
        )

        path = blend_out_path.with_name(
            f"{blend_out_path.stem}_alpha{alpha_name}{blend_out_path.suffix}"
        )

        LOGGER.info(
            "Writing alpha overlay: %s (alpha=%.2f)",
            path,
            alpha,
        )

        if not cv2.imwrite(
            str(path),
            output,
        ):
            raise RuntimeError(
                f"Could not write overlay image: {path}",
            )

    LOGGER.info(
        "Export completed successfully.",
    )


# ============================================================================
# Automatic initialization: ORB
# ============================================================================


def auto_init_points(
    vis_full: np.ndarray,
    sem_full: np.ndarray,
    max_points: int = AUTO_INIT_MAX_POINTS,
) -> tuple[list[Point], list[Point]]:
    """
    Estimate VIS <-> SEM correspondences using ORB.

    Pair integrity is preserved during spatial deduplication.
    """

    LOGGER.info(
        "Starting ORB auto-initialization.",
    )

    vis_gray = cv2.cvtColor(
        vis_full,
        cv2.COLOR_BGR2GRAY,
    )

    sem_gray = cv2.cvtColor(
        sem_full,
        cv2.COLOR_BGR2GRAY,
    )

    orb = cv2.ORB_create(
        nfeatures=ORB_FEATURE_COUNT,
    )

    keypoints_vis, descriptors_vis = orb.detectAndCompute(
        vis_gray,
        None,
    )

    keypoints_sem, descriptors_sem = orb.detectAndCompute(
        sem_gray,
        None,
    )

    LOGGER.info(
        "ORB features: VIS=%d SEM=%d",
        len(keypoints_vis),
        len(keypoints_sem),
    )

    if descriptors_vis is None or descriptors_sem is None:
        LOGGER.warning(
            "ORB descriptors unavailable.",
        )
        return [], []

    if not keypoints_vis or not keypoints_sem:
        LOGGER.warning(
            "ORB found no usable keypoints.",
        )
        return [], []

    matcher = cv2.BFMatcher(
        cv2.NORM_HAMMING,
        crossCheck=True,
    )

    matches = list(
        matcher.match(
            descriptors_sem,
            descriptors_vis,
        )
    )

    matches.sort(
        key=lambda match: match.distance,
    )

    LOGGER.info(
        "ORB raw matches: %d",
        len(matches),
    )

    if not matches:
        return [], []

    LOGGER.debug(
        "Best ORB match distance: %.2f",
        matches[0].distance,
    )

    # Keep the entire correspondence together.
    candidates: list[tuple[Point, Point, float]] = []

    for match in matches:
        sem_point = tuple(round(value) for value in keypoints_sem[match.queryIdx].pt)

        vis_point = tuple(round(value) for value in keypoints_vis[match.trainIdx].pt)

        candidates.append((
            vis_point,
            sem_point,
            float(match.distance),
        ))

    selected: list[tuple[Point, Point]] = []

    min_distance_sq = ORB_SPATIAL_DEDUP_PX**2

    for vis_point, sem_point, distance in candidates:
        del distance

        duplicate = False

        for selected_vis, selected_sem in selected:
            vis_dx = vis_point[0] - selected_vis[0]
            vis_dy = vis_point[1] - selected_vis[1]

            sem_dx = sem_point[0] - selected_sem[0]
            sem_dy = sem_point[1] - selected_sem[1]

            vis_distance_sq = vis_dx * vis_dx + vis_dy * vis_dy
            sem_distance_sq = sem_dx * sem_dx + sem_dy * sem_dy

            if vis_distance_sq < min_distance_sq or sem_distance_sq < min_distance_sq:
                duplicate = True
                break

        if duplicate:
            continue

        selected.append((
            vis_point,
            sem_point,
        ))

        if len(selected) >= max_points:
            break

    vis_points = [vis_point for vis_point, _ in selected]

    sem_points = [sem_point for _, sem_point in selected]

    LOGGER.info(
        "ORB initialization result: %d point pairs",
        len(selected),
    )

    LOGGER.debug(
        "ORB VIS points: %s",
        vis_points,
    )

    LOGGER.debug(
        "ORB SEM points: %s",
        sem_points,
    )

    return vis_points, sem_points


# ============================================================================
# Automatic initialization: boundary
# ============================================================================


def _largest_contour(
    image: np.ndarray,
) -> np.ndarray | None:
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    )

    _, binary = cv2.threshold(
        gray,
        BOUNDARY_THRESHOLD,
        255,
        cv2.THRESH_BINARY,
    )

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    if not contours:
        return None

    image_area = image.shape[0] * image.shape[1]

    valid = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= image_area * BOUNDARY_MIN_AREA_FRACTION
    ]

    if not valid:
        return None

    return max(
        valid,
        key=cv2.contourArea,
    )


def _boundary_points(
    image: np.ndarray,
    max_corners: int,
) -> list[Point]:
    contour = _largest_contour(
        image,
    )

    if contour is None:
        LOGGER.warning(
            "No suitable boundary contour found.",
        )
        return []

    perimeter = cv2.arcLength(
        contour,
        True,
    )

    epsilon = 0.02 * perimeter

    polygon = cv2.approxPolyDP(
        contour,
        epsilon,
        True,
    )

    points = [
        (
            int(point[0][0]),
            int(point[0][1]),
        )
        for point in polygon
    ]

    if len(points) < 4:
        LOGGER.warning(
            "Boundary polygon has fewer than 4 vertices.",
        )
        return []

    center_x = sum(point[0] for point in points) / len(points)

    center_y = sum(point[1] for point in points) / len(points)

    points.sort(
        key=lambda point: np.arctan2(
            point[1] - center_y,
            point[0] - center_x,
        )
    )

    return points[:max_corners]


def auto_init_points_boundary(
    vis_full: np.ndarray,
    sem_full: np.ndarray,
    max_corners: int = BOUNDARY_MAX_CORNERS,
) -> tuple[list[Point], list[Point]]:
    LOGGER.info(
        "Starting boundary auto-initialization.",
    )

    vis_points = _boundary_points(
        vis_full,
        max_corners,
    )

    sem_points = _boundary_points(
        sem_full,
        max_corners,
    )

    count = min(
        len(vis_points),
        len(sem_points),
        max_corners,
    )

    if count < 4:
        LOGGER.warning(
            "Boundary initialization produced fewer than 4 usable point pairs.",
        )
        return [], []

    result_vis = vis_points[:count]
    result_sem = sem_points[:count]

    LOGGER.info(
        "Boundary initialization result: %d point pairs",
        count,
    )

    return result_vis, result_sem


# ============================================================================
# GUI
# ============================================================================


class SemVisApp:
    """Three-pane Tkinter GUI with synced zoom and fit-to-width."""

    def __init__(
        self,
        root: tk.Tk,
        vis_full: np.ndarray,
        sem_full: np.ndarray,
        state: RegistrationState,
        display_scale: float,
        alpha: float,
        vis_path: Path,
        sem_path: Path,
    ) -> None:
        self._root = root
        self._vis_full = vis_full
        self._sem_full = sem_full
        self._state = state
        self._alpha = alpha
        self._vis_path = vis_path
        self._sem_path = sem_path

        self._root.title("semvis - Registration")

        # Configure window size
        screen_w = self._root.winfo_screenwidth()
        screen_h = self._root.winfo_screenheight()
        win_w = max(960, screen_w - 40)
        win_h = max(600, screen_h - 100)
        self._root.geometry(f"{win_w}x{win_h}+10+10")
        self._root.update_idletasks()

        # Fit images to window pane width on start
        pane_w = max(200, (win_w - 30) // 3)
        self._base_scale_vis = pane_w / vis_full.shape[1]
        self._base_scale_sem = pane_w / sem_full.shape[1]

        self._zoom: float = 1.0
        self._offsets: dict[str, list[float]] = {
            "vis": [0.0, 0.0],
            "sem": [0.0, 0.0],
            "preview": [0.0, 0.0],
        }

        self._pan_start_x: int | None = None
        self._pan_start_y: int | None = None

        # Keep PhotoImage references to prevent garbage collection
        self._vis_photo: ImageTk.PhotoImage | None = None
        self._sem_photo: ImageTk.PhotoImage | None = None
        self._preview_photo: ImageTk.PhotoImage | None = None

        self._create_widgets()
        self._bind_events()
        self._refresh()

    def _create_widgets(self) -> None:
        # Toolbar
        toolbar = ttk.Frame(self._root, padding=4)
        toolbar.pack(side=tk.TOP, fill=tk.X)

        ttk.Button(toolbar, text="Fit Width (0)", command=self._cmd_fit).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="ORB (A)", command=self._cmd_orb).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Boundary (B)", command=self._cmd_boundary).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Clear (C)", command=self._cmd_clear).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Delete Last (D)", command=self._cmd_delete).pack(side=tk.LEFT, padx=2)
        ttk.Separator(toolbar, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=6)
        ttk.Button(toolbar, text="Save (S)", command=self._cmd_save).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Export (E)", command=self._cmd_export).pack(side=tk.LEFT, padx=2)
        ttk.Button(toolbar, text="Quit (ESC)", command=self._cmd_quit).pack(side=tk.RIGHT, padx=2)

        # Status Bar
        self._status_var = tk.StringVar()
        statusbar = ttk.Label(
            self._root,
            textvariable=self._status_var,
            relief=tk.SUNKEN,
            anchor=tk.W,
            padding=(6, 3),
        )
        statusbar.pack(side=tk.BOTTOM, fill=tk.X)

        # Panes container
        panes = ttk.PanedWindow(self._root, orient=tk.HORIZONTAL)
        panes.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=4, pady=4)

        # VIS Pane
        vis_frame = ttk.LabelFrame(panes, text="VIS", padding=2)
        panes.add(vis_frame, weight=1)
        self._vis_canvas = tk.Canvas(
            vis_frame,
            bg="black",
            highlightthickness=0,
        )
        self._vis_canvas.pack(fill=tk.BOTH, expand=True)
        self._vis_img_id = self._vis_canvas.create_image(0, 0, anchor=tk.NW)

        # SEM Pane
        sem_frame = ttk.LabelFrame(panes, text="SEM", padding=2)
        panes.add(sem_frame, weight=1)
        self._sem_canvas = tk.Canvas(
            sem_frame,
            bg="black",
            highlightthickness=0,
        )
        self._sem_canvas.pack(fill=tk.BOTH, expand=True)
        self._sem_img_id = self._sem_canvas.create_image(0, 0, anchor=tk.NW)

        # Preview Pane
        preview_frame = ttk.LabelFrame(panes, text="SEM + warped VIS", padding=2)
        panes.add(preview_frame, weight=1)
        self._preview_canvas = tk.Canvas(
            preview_frame,
            bg="black",
            highlightthickness=0,
        )
        self._preview_canvas.pack(fill=tk.BOTH, expand=True)
        self._preview_img_id = self._preview_canvas.create_image(0, 0, anchor=tk.NW)

    def _bind_events(self) -> None:
        # Mouse bindings for VIS canvas
        self._vis_canvas.bind("<ButtonPress-1>", self._vis_press)
        self._vis_canvas.bind("<B1-Motion>", self._vis_motion)
        self._vis_canvas.bind("<ButtonRelease-1>", self._vis_release)

        # Mouse bindings for SEM canvas
        self._sem_canvas.bind("<ButtonPress-1>", self._sem_press)
        self._sem_canvas.bind("<B1-Motion>", self._sem_motion)
        self._sem_canvas.bind("<ButtonRelease-1>", self._sem_release)

        # Synced Zoom and Pan on all canvases
        for canvas in (self._vis_canvas, self._sem_canvas, self._preview_canvas):
            # Linux scroll wheel
            canvas.bind("<Control-Button-4>", lambda e: self._on_zoom(e, 1.15))
            canvas.bind("<Control-Button-5>", lambda e: self._on_zoom(e, 1.0 / 1.15))
            # Windows / macOS / Tk wheel
            canvas.bind("<Control-MouseWheel>", self._on_mousewheel)

            # Panning with middle mouse button
            canvas.bind("<ButtonPress-2>", self._pan_start)
            canvas.bind("<B2-Motion>", self._pan_move)
            canvas.bind("<ButtonRelease-2>", self._pan_end)

            # Panning with right mouse button
            canvas.bind("<ButtonPress-3>", self._pan_start)
            canvas.bind("<B3-Motion>", self._pan_move)
            canvas.bind("<ButtonRelease-3>", self._pan_end)

        # Global key shortcuts
        self._root.bind("<Escape>", lambda _: self._cmd_quit())
        self._root.bind("<KeyPress-0>", lambda _: self._cmd_fit())
        self._root.bind("<KeyPress-r>", lambda _: self._cmd_fit())
        self._root.bind("<KeyPress-R>", lambda _: self._cmd_fit())
        self._root.bind("<KeyPress-a>", lambda _: self._cmd_orb())
        self._root.bind("<KeyPress-A>", lambda _: self._cmd_orb())
        self._root.bind("<KeyPress-b>", lambda _: self._cmd_boundary())
        self._root.bind("<KeyPress-B>", lambda _: self._cmd_boundary())
        self._root.bind("<KeyPress-c>", lambda _: self._cmd_clear())
        self._root.bind("<KeyPress-C>", lambda _: self._cmd_clear())
        self._root.bind("<KeyPress-d>", lambda _: self._cmd_delete())
        self._root.bind("<KeyPress-D>", lambda _: self._cmd_delete())
        self._root.bind("<KeyPress-s>", lambda _: self._cmd_save())
        self._root.bind("<KeyPress-S>", lambda _: self._cmd_save())
        self._root.bind("<KeyPress-e>", lambda _: self._cmd_export())
        self._root.bind("<KeyPress-E>", lambda _: self._cmd_export())

    def _on_zoom(self, event: tk.Event, factor: float) -> None:
        mx = event.x
        my = event.y

        old_zoom = self._zoom
        new_zoom = max(0.05, min(50.0, old_zoom * factor))
        if abs(new_zoom - old_zoom) < 1e-6:
            return

        actual_factor = new_zoom / old_zoom
        self._zoom = new_zoom

        # Keep same pixel under mouse position across all 3 synced views
        for view_name in ("vis", "sem", "preview"):
            ox, oy = self._offsets[view_name]
            self._offsets[view_name][0] = mx - (mx - ox) * actual_factor
            self._offsets[view_name][1] = my - (my - oy) * actual_factor

        self._refresh()

    def _on_mousewheel(self, event: tk.Event) -> None:
        if event.delta > 0:
            self._on_zoom(event, 1.15)
        elif event.delta < 0:
            self._on_zoom(event, 1.0 / 1.15)

    def _pan_start(self, event: tk.Event) -> None:
        self._pan_start_x = event.x
        self._pan_start_y = event.y

    def _pan_move(self, event: tk.Event) -> None:
        if self._pan_start_x is not None and self._pan_start_y is not None:
            dx = event.x - self._pan_start_x
            dy = event.y - self._pan_start_y
            self._pan_start_x = event.x
            self._pan_start_y = event.y

            for view_name in ("vis", "sem", "preview"):
                self._offsets[view_name][0] += dx
                self._offsets[view_name][1] += dy

            self._refresh()

    def _pan_end(self, event: tk.Event) -> None:
        del event
        self._pan_start_x = None
        self._pan_start_y = None

    def _cmd_fit(self) -> None:
        self._zoom = 1.0
        for key in self._offsets:
            self._offsets[key] = [0.0, 0.0]
        self._refresh()

    def _numpy_to_photoimage(self, bgr: np.ndarray) -> ImageTk.PhotoImage:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        return ImageTk.PhotoImage(pil)

    def _update_status(self) -> None:
        valid_str = (
            "✓ valid (ready to save/export)"
            if self._state.valid()
            else "✗ need at least 4 matching point pairs"
        )
        zoom_pct = round(self._zoom * 100)
        self._status_var.set(
            f"Zoom: {zoom_pct}% | "
            f"VIS points: {len(self._state.vis)} | "
            f"SEM points: {len(self._state.sem)} | "
            f"Pairs: {self._state.pair_count()} | "
            f"{valid_str} | "
            f"[Ctrl+Wheel: Zoom | Mid/Right drag: Pan]"
        )

    def _refresh(self) -> None:
        LOGGER.debug(
            "Refreshing GUI: VIS=%d SEM=%d pairs=%d valid=%s zoom=%.2f",
            len(self._state.vis),
            len(self._state.sem),
            self._state.pair_count(),
            self._state.valid(),
            self._zoom,
        )

        scale_vis = self._base_scale_vis * self._zoom
        scale_sem = self._base_scale_sem * self._zoom

        vis_display = resize_for_display(
            self._vis_full,
            scale_vis,
        )
        sem_display = resize_for_display(
            self._sem_full,
            scale_sem,
        )

        vis_window = draw_points(
            vis_display,
            self._state.vis,
            scale_vis,
        )

        sem_window = draw_points(
            sem_display,
            self._state.sem,
            scale_sem,
        )

        preview = preview_overlay(
            vis_display,
            sem_display,
            self._state,
            scale_vis,
            self._alpha,
            sem_scale=scale_sem,
        )

        self._vis_photo = self._numpy_to_photoimage(vis_window)
        self._sem_photo = self._numpy_to_photoimage(sem_window)
        self._preview_photo = self._numpy_to_photoimage(preview)

        self._vis_canvas.itemconfig(self._vis_img_id, image=self._vis_photo)
        self._vis_canvas.coords(self._vis_img_id, self._offsets["vis"][0], self._offsets["vis"][1])

        self._sem_canvas.itemconfig(self._sem_img_id, image=self._sem_photo)
        self._sem_canvas.coords(self._sem_img_id, self._offsets["sem"][0], self._offsets["sem"][1])

        self._preview_canvas.itemconfig(self._preview_img_id, image=self._preview_photo)
        self._preview_canvas.coords(self._preview_img_id, self._offsets["preview"][0], self._offsets["preview"][1])

        self._update_status()

    # ------------------------------------------------------------------------
    # VIS mouse
    # ------------------------------------------------------------------------

    def _vis_press(self, event: tk.Event) -> None:
        scale = self._base_scale_vis * self._zoom
        offset = (self._offsets["vis"][0], self._offsets["vis"][1])

        index = nearest_point_index(
            self._state.vis,
            (event.x, event.y),
            scale,
            offset=offset,
        )

        if index is not None:
            LOGGER.debug(
                "VIS drag start: point %d",
                index + 1,
            )
            self._state.drag_idx = index
            self._state.drag_mode = "vis"
        else:
            point = canvas_to_full_point(
                (event.x, event.y),
                scale,
                offset=offset,
            )
            if 0 <= point[0] < self._vis_full.shape[1] and 0 <= point[1] < self._vis_full.shape[0]:
                LOGGER.info(
                    "Adding VIS point %d: %s",
                    len(self._state.vis) + 1,
                    point,
                )
                self._state.vis.append(point)

        self._refresh()

    def _vis_motion(self, event: tk.Event) -> None:
        if self._state.drag_idx is not None and self._state.drag_mode == "vis":
            scale = self._base_scale_vis * self._zoom
            offset = (self._offsets["vis"][0], self._offsets["vis"][1])

            point = canvas_to_full_point(
                (event.x, event.y),
                scale,
                offset=offset,
            )
            self._state.vis[self._state.drag_idx] = point
            self._refresh()

    def _vis_release(self, event: tk.Event) -> None:
        del event
        if self._state.drag_idx is not None:
            LOGGER.debug(
                "VIS drag end: point %d",
                self._state.drag_idx + 1,
            )
        self._state.drag_idx = None
        self._state.drag_mode = None

    # ------------------------------------------------------------------------
    # SEM mouse
    # ------------------------------------------------------------------------

    def _sem_press(self, event: tk.Event) -> None:
        scale = self._base_scale_sem * self._zoom
        offset = (self._offsets["sem"][0], self._offsets["sem"][1])

        index = nearest_point_index(
            self._state.sem,
            (event.x, event.y),
            scale,
            offset=offset,
        )

        if index is not None:
            LOGGER.debug(
                "SEM drag start: point %d",
                index + 1,
            )
            self._state.drag_idx = index
            self._state.drag_mode = "sem"
        else:
            point = canvas_to_full_point(
                (event.x, event.y),
                scale,
                offset=offset,
            )
            if 0 <= point[0] < self._sem_full.shape[1] and 0 <= point[1] < self._sem_full.shape[0]:
                LOGGER.info(
                    "Adding SEM point %d: (%d,%d)",
                    len(self._state.sem) + 1,
                    point[0],
                    point[1],
                )
                self._state.sem.append(point)

        self._refresh()

    def _sem_motion(self, event: tk.Event) -> None:
        if self._state.drag_idx is not None and self._state.drag_mode == "sem":
            scale = self._base_scale_sem * self._zoom
            offset = (self._offsets["sem"][0], self._offsets["sem"][1])

            point = canvas_to_full_point(
                (event.x, event.y),
                scale,
                offset=offset,
            )
            self._state.sem[self._state.drag_idx] = point
            self._refresh()

    def _sem_release(self, event: tk.Event) -> None:
        del event
        if self._state.drag_idx is not None:
            LOGGER.debug(
                "SEM drag end: point %d",
                self._state.drag_idx + 1,
            )
        self._state.drag_idx = None
        self._state.drag_mode = None

    # ------------------------------------------------------------------------
    # Commands / actions
    # ------------------------------------------------------------------------

    def _cmd_quit(self) -> None:
        LOGGER.info(
            "Quitting GUI.",
        )
        self._root.destroy()

    def _cmd_delete(self) -> None:
        LOGGER.info(
            "Deleting last point.",
        )
        if self._state.vis:
            removed = self._state.vis.pop()
            LOGGER.debug(
                "Removed VIS point: %s",
                removed,
            )
        if self._state.sem:
            removed = self._state.sem.pop()
            LOGGER.debug(
                "Removed SEM point: %s",
                removed,
            )
        LOGGER.info(
            "Points after delete: VIS=%d SEM=%d",
            len(self._state.vis),
            len(self._state.sem),
        )
        self._refresh()

    def _cmd_clear(self) -> None:
        LOGGER.info(
            "Clearing all points.",
        )
        self._state.vis.clear()
        self._state.sem.clear()
        self._refresh()

    def _cmd_orb(self) -> None:
        LOGGER.info(
            "Starting ORB initialization.",
        )
        vis_points, sem_points = auto_init_points(
            self._vis_full,
            self._sem_full,
        )
        if len(vis_points) >= 4:
            self._state.vis = vis_points
            self._state.sem = sem_points
            LOGGER.info(
                "ORB initialization accepted: %d pairs",
                len(vis_points),
            )
        else:
            LOGGER.warning(
                "ORB initialization rejected: %d pairs",
                len(vis_points),
            )
            messagebox.showwarning(
                "ORB Initialization",
                f"ORB found fewer than 4 usable point pairs ({len(vis_points)}).",
            )
        self._refresh()

    def _cmd_boundary(self) -> None:
        LOGGER.info(
            "Starting boundary initialization.",
        )
        vis_points, sem_points = auto_init_points_boundary(
            self._vis_full,
            self._sem_full,
        )
        if len(vis_points) >= 4:
            self._state.vis = vis_points
            self._state.sem = sem_points
            LOGGER.info(
                "Boundary initialization accepted: %d pairs",
                len(vis_points),
            )
        else:
            LOGGER.warning(
                "Boundary initialization rejected: %d pairs",
                len(vis_points),
            )
            messagebox.showwarning(
                "Boundary Initialization",
                f"Boundary initialization produced fewer than 4 usable pairs ({len(vis_points)}).",
            )
        self._refresh()

    def _cmd_save(self) -> None:
        LOGGER.info(
            "Save requested.",
        )
        if not self._state.valid():
            msg = (
                f"Cannot save: VIS={len(self._state.vis)} SEM={len(self._state.sem)}. "
                "Need equal counts and at least 4 pairs."
            )
            LOGGER.warning(msg)
            messagebox.showwarning(
                "Save Registration",
                msg,
            )
            return

        try:
            registration_file = registration_path(
                self._sem_path,
                self._vis_path,
            )
            save_registration(
                registration_file,
                self._state.registration(),
            )
            LOGGER.info(
                "Registration saved: %s",
                registration_file,
            )
            messagebox.showinfo(
                "Save Registration",
                f"Registration saved successfully:\n{registration_file}",
            )
        except Exception as exc:
            LOGGER.exception(
                "Could not save registration.",
            )
            messagebox.showerror(
                "Save Error",
                f"Could not save registration:\n{exc}",
            )

    def _cmd_export(self) -> None:
        LOGGER.info(
            "Export requested.",
        )
        if not self._state.valid():
            msg = (
                f"Cannot export: VIS={len(self._state.vis)} SEM={len(self._state.sem)}. "
                "Need equal counts and at least 4 pairs."
            )
            LOGGER.warning(msg)
            messagebox.showwarning(
                "Export",
                msg,
            )
            return

        try:
            warp_path, blend_path = default_export_paths(
                self._sem_path,
                self._vis_path,
            )
            export_full(
                self._vis_full,
                self._sem_full,
                self._state,
                warp_path,
                blend_path,
                self._alpha,
            )
            registration_file = registration_path(
                self._sem_path,
                self._vis_path,
            )
            save_registration(
                registration_file,
                self._state.registration(),
            )
            LOGGER.info(
                "Registration saved: %s",
                registration_file,
            )
            LOGGER.info(
                "Warped VIS saved: %s",
                warp_path,
            )
            LOGGER.info(
                "Overlay saved: %s",
                blend_path,
            )
            messagebox.showinfo(
                "Export Successful",
                f"Export completed successfully!\n\n"
                f"Registration: {registration_file.name}\n"
                f"Warped VIS: {warp_path.name}\n"
                f"Overlay: {blend_path.name}",
            )
            self._root.destroy()
        except Exception as exc:
            LOGGER.exception(
                "Export failed.",
            )
            messagebox.showerror(
                "Export Error",
                f"Export failed:\n{exc}",
            )

    def run(self) -> None:
        self._root.mainloop()


def semvis(
    vis_full: np.ndarray,
    sem_full: np.ndarray,
    state: RegistrationState,
    display_scale: float,
    alpha: float,
    vis_path: Path,
    sem_path: Path,
) -> None:
    LOGGER.info(
        "Starting GUI. VIS=%d SEM=%d pairs=%d",
        len(state.vis),
        len(state.sem),
        state.pair_count(),
    )

    root = tk.Tk()
    app = SemVisApp(
        root=root,
        vis_full=vis_full,
        sem_full=sem_full,
        state=state,
        display_scale=display_scale,
        alpha=alpha,
        vis_path=vis_path,
        sem_path=sem_path,
    )
    app.run()

    LOGGER.info(
        "GUI stopped.",
    )


# ============================================================================
# Output paths
# ============================================================================


def default_export_paths(
    sem_path: Path,
    vis_path: Path,
) -> tuple[Path, Path]:
    directory = sem_path.parent

    warped = directory / (f"{sem_path.stem}_{vis_path.stem}_vis_warped{vis_path.suffix}")

    overlay = directory / (f"{sem_path.stem}_{vis_path.stem}_overlay{sem_path.suffix}")

    return warped, overlay


# ============================================================================
# CLI
# ============================================================================


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Register VIS to SEM. SEM is fixed and VIS is warped into SEM coordinates."),
    )

    parser.add_argument(
        "--vis",
        "--opt",
        dest="vis",
        type=Path,
        default=DEFAULT_VIS_PATH,
        help="VIS image path.",
    )

    parser.add_argument(
        "--sem",
        type=Path,
        default=DEFAULT_SEM_PATH,
        help="SEM reference image path.",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=VIS_DISPLAY_SCALE,
        help="VIS GUI display scale.",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=ALPHA,
        help="Preview/export overlay alpha.",
    )

    parser.add_argument(
        "--log-level",
        choices=(
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ),
        default="INFO",
        help="Logging verbosity.",
    )

    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Do not open the GUI.",
    )

    parser.add_argument(
        "--export",
        action="store_true",
        help="Export using the saved registration.",
    )

    parser.add_argument(
        "--warp-out",
        type=Path,
        default=None,
        help="Output path for warped VIS.",
    )

    parser.add_argument(
        "--blend-out",
        type=Path,
        default=None,
        help="Output path for SEM/VIS overlay.",
    )

    return parser


def run_export(
    vis_path: Path,
    sem_path: Path,
    alpha: float,
    warp_out: Path | None,
    blend_out: Path | None,
) -> None:
    LOGGER.info(
        "Running headless export.",
    )

    vis_full = load_image(
        vis_path,
    )

    sem_full = load_image(
        sem_path,
    )

    registration_file = registration_path(
        sem_path,
        vis_path,
    )

    registration = load_registration(
        registration_file,
    )

    if not registration.valid():
        raise RuntimeError(
            "Saved registration does not contain at least 4 matching point pairs.",
        )

    state = RegistrationState(
        vis=list(registration.vis),
        sem=list(registration.sem),
    )

    default_warp, default_blend = default_export_paths(
        sem_path,
        vis_path,
    )

    export_full(
        vis_full,
        sem_full,
        state,
        warp_out or default_warp,
        blend_out or default_blend,
        alpha,
    )

    LOGGER.info(
        "Headless export finished.",
    )


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    configure_logging(
        args.log_level,
    )

    if args.scale <= 0:
        parser.error(
            "--scale must be > 0.",
        )

    if not 0.0 <= args.alpha <= 1.0:
        parser.error(
            "--alpha must be between 0 and 1.",
        )

    vis_path: Path = args.vis
    sem_path: Path = args.sem

    LOGGER.info(
        "VIS: %s",
        vis_path,
    )

    LOGGER.info(
        "SEM: %s",
        sem_path,
    )

    if not vis_path.exists():
        parser.error(
            f"VIS image does not exist: {vis_path}",
        )

    if not sem_path.exists():
        parser.error(
            f"SEM image does not exist: {sem_path}",
        )

    try:
        vis_full = load_image(
            vis_path,
        )

        sem_full = load_image(
            sem_path,
        )

        registration_file = registration_path(
            sem_path,
            vis_path,
        )

        # ------------------------------------------------------------
        # Headless export
        # ------------------------------------------------------------

        if args.export or args.no_gui:
            if not registration_file.exists():
                raise RuntimeError(
                    f"Registration file not found: {registration_file}",
                )

            run_export(
                vis_path,
                sem_path,
                args.alpha,
                args.warp_out,
                args.blend_out,
            )

            return 0

        # ------------------------------------------------------------
        # Load existing registration
        # ------------------------------------------------------------

        if registration_file.exists():
            try:
                registration = load_registration(
                    registration_file,
                )

                state = RegistrationState(
                    vis=list(registration.vis),
                    sem=list(registration.sem),
                )

                LOGGER.info(
                    "Using saved registration: %d point pairs",
                    len(state.vis),
                )

            except (
                OSError,
                ValueError,
                json.JSONDecodeError,
            ):
                LOGGER.exception(
                    "Could not load saved registration. Falling back to ORB.",
                )

                vis_points, sem_points = auto_init_points(
                    vis_full,
                    sem_full,
                )

                state = RegistrationState(
                    vis=vis_points,
                    sem=sem_points,
                )

        # ------------------------------------------------------------
        # No registration: ORB initialization
        # ------------------------------------------------------------

        else:
            LOGGER.info(
                "No saved registration found: %s",
                registration_file,
            )

            vis_points, sem_points = auto_init_points(
                vis_full,
                sem_full,
            )

            state = RegistrationState(
                vis=vis_points,
                sem=sem_points,
            )

            LOGGER.info(
                "ORB initialized %d point pairs.",
                len(state.vis),
            )

        # ------------------------------------------------------------
        # GUI
        # ------------------------------------------------------------

        semvis(
            vis_full,
            sem_full,
            state,
            args.scale,
            args.alpha,
            vis_path,
            sem_path,
        )

        return 0

    except Exception:
        LOGGER.exception(
            "Fatal error.",
        )

        return 1


if __name__ == "__main__":
    raise SystemExit(
        main(),
    )
