#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

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


def screen_size() -> tuple[int, int]:
    root = tk.Tk()
    root.withdraw()

    width = root.winfo_screenwidth()
    height = root.winfo_screenheight()

    root.destroy()

    return width, height


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


def nearest_point_index(
    points: list[Point],
    display_point: Point,
    display_scale: float,
    radius: float = DRAG_HIT_RADIUS_PX,
) -> int | None:
    if not points:
        return None

    best_index: int | None = None
    best_distance_sq = radius * radius

    x, y = display_point

    for index, point in enumerate(points):
        px, py = full_to_display_point(
            point,
            display_scale,
        )

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
    display_scale: float,
) -> np.ndarray | None:
    """
    Convert VIS(full) -> SEM(full) into:

        VIS(display) -> SEM(full)

    SEM is not scaled.
    """

    homography = homography_full(
        state,
    )

    if homography is None:
        return None

    scale_inverse = np.array(
        [
            [1.0 / display_scale, 0.0, 0.0],
            [0.0, 1.0 / display_scale, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    return homography @ scale_inverse


# ============================================================================
# Preview / export
# ============================================================================


def preview_overlay(
    vis_display: np.ndarray,
    sem_display: np.ndarray,
    state: RegistrationState,
    display_scale: float,
    alpha: float,
) -> np.ndarray:
    homography = homography_display(
        state,
        display_scale,
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


WINDOW_VIS = "VIS"
WINDOW_SEM = "SEM"
WINDOW_PREVIEW = "SEM + warped VIS"


def _draw_vis_window(
    vis_display: np.ndarray,
    state: RegistrationState,
    display_scale: float,
) -> np.ndarray:
    return draw_points(
        vis_display,
        state.vis,
        display_scale,
    )


def _draw_sem_window(
    sem_display: np.ndarray,
    state: RegistrationState,
) -> np.ndarray:
    return draw_points(
        sem_display,
        state.sem,
        1.0,
    )


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

    vis_display = resize_for_display(
        vis_full,
        display_scale,
    )

    sem_display = sem_full.copy()

    screen_w, screen_h = screen_size()

    LOGGER.info(
        "Screen size: %dx%d",
        screen_w,
        screen_h,
    )

    # Three windows side-by-side.
    # Each gets one third of the available width.
    window_w = max(
        320,
        screen_w // 3,
    )

    # Height is chosen to fit the screen while preserving enough room
    # for the largest image.
    window_h = max(
        240,
        screen_h - 80,
    )

    LOGGER.info(
        "GUI window size: %dx%d",
        window_w,
        window_h,
    )

    cv2.namedWindow(
        WINDOW_VIS,
        cv2.WINDOW_NORMAL,
    )

    cv2.namedWindow(
        WINDOW_SEM,
        cv2.WINDOW_NORMAL,
    )

    cv2.namedWindow(
        WINDOW_PREVIEW,
        cv2.WINDOW_NORMAL,
    )

    cv2.resizeWindow(
        WINDOW_VIS,
        window_w,
        window_h,
    )

    cv2.resizeWindow(
        WINDOW_SEM,
        window_w,
        window_h,
    )

    cv2.resizeWindow(
        WINDOW_PREVIEW,
        window_w,
        window_h,
    )

    cv2.moveWindow(
        WINDOW_VIS,
        0,
        0,
    )

    cv2.moveWindow(
        WINDOW_SEM,
        window_w,
        0,
    )

    cv2.moveWindow(
        WINDOW_PREVIEW,
        2 * window_w,
        0,
    )

    def refresh() -> None:
        LOGGER.debug(
            "Refreshing GUI: VIS=%d SEM=%d pairs=%d valid=%s",
            len(state.vis),
            len(state.sem),
            state.pair_count(),
            state.valid(),
        )

        vis_window = _draw_vis_window(
            vis_display,
            state,
            display_scale,
        )

        sem_window = _draw_sem_window(
            sem_display,
            state,
        )

        preview = preview_overlay(
            vis_display,
            sem_display,
            state,
            display_scale,
            alpha,
        )

        cv2.imshow(
            WINDOW_VIS,
            vis_window,
        )

        cv2.imshow(
            WINDOW_SEM,
            sem_window,
        )

        cv2.imshow(
            WINDOW_PREVIEW,
            preview,
        )

    # ------------------------------------------------------------------------
    # VIS mouse
    # ------------------------------------------------------------------------

    def vis_mouse(
        event: int,
        x: int,
        y: int,
        flags: int,
        userdata: object,
    ) -> None:
        del userdata

        if event == cv2.EVENT_LBUTTONDOWN:
            index = nearest_point_index(
                state.vis,
                (x, y),
                display_scale,
            )

            if index is not None:
                LOGGER.debug(
                    "VIS drag start: point %d",
                    index + 1,
                )

                state.drag_idx = index
                state.drag_mode = "vis"

            else:
                point = display_to_full_point(
                    (x, y),
                    display_scale,
                )

                LOGGER.info(
                    "Adding VIS point %d: %s",
                    len(state.vis) + 1,
                    point,
                )

                # IMPORTANT:
                # Do NOT add a placeholder to SEM.
                #
                # The two images are independent while points are being
                # created. Pairing happens by list index once both sides
                # contain the same number of points.
                state.vis.append(point)

            refresh()

        elif (
            event == cv2.EVENT_MOUSEMOVE
            and state.drag_idx is not None
            and state.drag_mode == "vis"
            and flags & cv2.EVENT_FLAG_LBUTTON
        ):
            point = display_to_full_point(
                (x, y),
                display_scale,
            )

            state.vis[state.drag_idx] = point

            refresh()

        elif event == cv2.EVENT_LBUTTONUP:
            if state.drag_idx is not None:
                LOGGER.debug(
                    "VIS drag end: point %d",
                    state.drag_idx + 1,
                )

            state.drag_idx = None
            state.drag_mode = None

    # ------------------------------------------------------------------------
    # SEM mouse
    # ------------------------------------------------------------------------

    def sem_mouse(
        event: int,
        x: int,
        y: int,
        flags: int,
        userdata: object,
    ) -> None:
        del userdata

        if event == cv2.EVENT_LBUTTONDOWN:
            index = nearest_point_index(
                state.sem,
                (x, y),
                1.0,
            )

            if index is not None:
                LOGGER.debug(
                    "SEM drag start: point %d",
                    index + 1,
                )

                state.drag_idx = index
                state.drag_mode = "sem"

            else:
                LOGGER.info(
                    "Adding SEM point %d: (%d,%d)",
                    len(state.sem) + 1,
                    x,
                    y,
                )

                # IMPORTANT:
                # Do NOT add a placeholder to VIS.
                state.sem.append(
                    (x, y),
                )

            refresh()

        elif (
            event == cv2.EVENT_MOUSEMOVE
            and state.drag_idx is not None
            and state.drag_mode == "sem"
            and flags & cv2.EVENT_FLAG_LBUTTON
        ):
            state.sem[state.drag_idx] = (
                x,
                y,
            )

            refresh()

        elif event == cv2.EVENT_LBUTTONUP:
            if state.drag_idx is not None:
                LOGGER.debug(
                    "SEM drag end: point %d",
                    state.drag_idx + 1,
                )

            state.drag_idx = None
            state.drag_mode = None

    cv2.setMouseCallback(
        WINDOW_VIS,
        vis_mouse,
    )

    cv2.setMouseCallback(
        WINDOW_SEM,
        sem_mouse,
    )

    refresh()

    LOGGER.info(
        "Controls: A=ORB B=boundary C=clear D=delete E=export S=save ESC=quit",
    )

    while True:
        key = cv2.waitKey(30) & 0xFF

        # ------------------------------------------------------------
        # Quit
        # ------------------------------------------------------------

        if key == 27:
            LOGGER.info(
                "ESC pressed; quitting.",
            )
            break

        # ------------------------------------------------------------
        # Delete last point
        # ------------------------------------------------------------

        elif key in (ord("d"), ord("D")):
            LOGGER.info(
                "D pressed: deleting last point.",
            )

            if state.vis:
                removed = state.vis.pop()

                LOGGER.debug(
                    "Removed VIS point: %s",
                    removed,
                )

            if state.sem:
                removed = state.sem.pop()

                LOGGER.debug(
                    "Removed SEM point: %s",
                    removed,
                )

            LOGGER.info(
                "Points after delete: VIS=%d SEM=%d",
                len(state.vis),
                len(state.sem),
            )

            refresh()

        # ------------------------------------------------------------
        # Clear
        # ------------------------------------------------------------

        elif key in (ord("c"), ord("C")):
            LOGGER.info(
                "C pressed: clearing all points.",
            )

            state.vis.clear()
            state.sem.clear()

            refresh()

        # ------------------------------------------------------------
        # ORB
        # ------------------------------------------------------------

        elif key in (ord("a"), ord("A")):
            LOGGER.info(
                "A pressed: starting ORB initialization.",
            )

            vis_points, sem_points = auto_init_points(
                vis_full,
                sem_full,
            )

            if len(vis_points) >= 4:
                state.vis = vis_points
                state.sem = sem_points

                LOGGER.info(
                    "ORB initialization accepted: %d pairs",
                    len(vis_points),
                )

            else:
                LOGGER.warning(
                    "ORB initialization rejected: %d pairs",
                    len(vis_points),
                )

            refresh()

        # ------------------------------------------------------------
        # Boundary
        # ------------------------------------------------------------

        elif key in (ord("b"), ord("B")):
            LOGGER.info(
                "B pressed: starting boundary initialization.",
            )

            vis_points, sem_points = auto_init_points_boundary(
                vis_full,
                sem_full,
            )

            if len(vis_points) >= 4:
                state.vis = vis_points
                state.sem = sem_points

                LOGGER.info(
                    "Boundary initialization accepted: %d pairs",
                    len(vis_points),
                )

            else:
                LOGGER.warning(
                    "Boundary initialization rejected: %d pairs",
                    len(vis_points),
                )

            refresh()

        # ------------------------------------------------------------
        # Save
        # ------------------------------------------------------------

        elif key in (ord("s"), ord("S")):
            LOGGER.info(
                "S pressed: save requested.",
            )

            if not state.valid():
                LOGGER.warning(
                    "Cannot save: VIS=%d SEM=%d. Need equal counts and at least 4 pairs.",
                    len(state.vis),
                    len(state.sem),
                )
                continue

            try:
                registration_file = registration_path(
                    sem_path,
                    vis_path,
                )

                save_registration(
                    registration_file,
                    state.registration(),
                )

                LOGGER.info(
                    "Registration saved: %s",
                    registration_file,
                )

            except Exception:
                LOGGER.exception(
                    "Could not save registration.",
                )

        # ------------------------------------------------------------
        # Export
        # ------------------------------------------------------------

        elif key in (ord("e"), ord("E")):
            LOGGER.info(
                "E pressed: export requested.",
            )

            if not state.valid():
                LOGGER.warning(
                    "Cannot export: VIS=%d SEM=%d. Need equal counts and at least 4 pairs.",
                    len(state.vis),
                    len(state.sem),
                )
                continue

            try:
                warp_path, blend_path = default_export_paths(
                    sem_path,
                    vis_path,
                )

                export_full(
                    vis_full,
                    sem_full,
                    state,
                    warp_path,
                    blend_path,
                    alpha,
                )

                registration_file = registration_path(
                    sem_path,
                    vis_path,
                )

                save_registration(
                    registration_file,
                    state.registration(),
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

                break

            except Exception:
                LOGGER.exception(
                    "Export failed.",
                )

    cv2.destroyAllWindows()

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
