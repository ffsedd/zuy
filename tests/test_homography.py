import numpy as np
import cv2
import pytest

from zuy.imoverlay.semvis import compute_full_homography

def test_homography_scale_translation():
    # Define a simple transform: scale by 2, translate by (10, 20)
    scale = 2.0
    tx = 10.0
    ty = 20.0

    # Source points (SEM space)
    src_pts = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
    ], dtype=np.float32)

    # Destination points (OPT space) after applying the transform
    dst_pts = np.array([
        [0.0 * scale + tx, 0.0 * scale + ty],
        [1.0 * scale + tx, 0.0 * scale + ty],
        [0.0 * scale + tx, 1.0 * scale + ty],
        [1.0 * scale + tx, 1.0 * scale + ty],
    ], dtype=np.float32)

    # Compute homography
    H_computed, _ = cv2.findHomography(src_pts, dst_pts, 0)

    # Expected homography matrix for pure scale and translation
    H_expected = np.array([
        [scale, 0.0, tx],
        [0.0, scale, ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    # Assert that the computed homography is close to the expected one
    np.testing.assert_allclose(H_computed, H_expected, atol=1e-6)


def test_compute_full_homography_scale_translation():
    # Similar setup to test_homography_scale_translation
    scale = 2.0
    tx = 10.0
    ty = 20.0
    display_scale_factor = 0.25 # Example display scale factor

    # Source points (SEM space) - full resolution
    src_pts_full = np.array([
        [0.0, 0.0],
        [100.0, 0.0],
        [0.0, 100.0],
        [100.0, 100.0],
    ], dtype=np.float32)

    # Destination points (OPT space) - full resolution after applying transform
    # The transform is (scale * x + tx, scale * y + ty)
    dst_pts_full = np.array([
        [0.0 * scale + tx, 0.0 * scale + ty],
        [100.0 * scale + tx, 0.0 * scale + ty],
        [0.0 * scale + tx, 100.0 * scale + ty],
        [100.0 * scale + tx, 100.0 * scale + ty],
    ], dtype=np.float32)

    # Destination points (OPT space) - display resolution
    dst_pts_disp = dst_pts_full * display_scale_factor

    # Compute homography at display resolution
    # H_disp maps src_pts_full -> dst_pts_disp
    H_disp, _ = cv2.findHomography(src_pts_full, dst_pts_disp, 0)

    # Compute full homography using the new function
    # H_full_computed should map src_pts_full -> dst_pts_full
    H_full_computed = compute_full_homography(H_disp, display_scale_factor)

    # Expected full homography for pure scale and translation
    H_expected_full = np.array([
        [scale, 0.0, tx],
        [0.0, scale, ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    np.testing.assert_allclose(H_full_computed, H_expected_full, atol=1e-6)
