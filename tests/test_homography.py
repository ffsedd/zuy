import numpy as np

from zuy.imoverlay.semvis import RegistrationState, homography_full


def test_homography_full_scale_translation() -> None:
    scale = 2.0
    tx = 10.0
    ty = 20.0

    vis_points = [
        (0.0, 0.0),
        (100.0, 0.0),
        (0.0, 100.0),
        (100.0, 100.0),
    ]

    sem_points = [
        (0.0 * scale + tx, 0.0 * scale + ty),
        (100.0 * scale + tx, 0.0 * scale + ty),
        (0.0 * scale + tx, 100.0 * scale + ty),
        (100.0 * scale + tx, 100.0 * scale + ty),
    ]

    state = RegistrationState(
        vis=vis_points,
        sem=sem_points,
    )

    H_computed = homography_full(state)

    assert H_computed is not None

    H_expected = np.array(
        [
            [scale, 0.0, tx],
            [0.0, scale, ty],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    # Homographies are defined only up to a non-zero scale.
    H_computed /= H_computed[2, 2]

    np.testing.assert_allclose(
        H_computed,
        H_expected,
        atol=1e-6,
    )
