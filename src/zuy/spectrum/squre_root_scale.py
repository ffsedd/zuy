from __future__ import annotations

from typing import Tuple

import matplotlib.pyplot as plt  # type: ignore
import numpy as np
from matplotlib.axis import Axis  # type: ignore
from matplotlib.scale import ScaleBase, register_scale  # type: ignore
from matplotlib.ticker import FuncFormatter, MaxNLocator  # type: ignore
from matplotlib.transforms import Transform  # type: ignore


class SqrtScale(ScaleBase):
    """
    Matplotlib scale that transforms axis values using the square root.
    Useful for compressing large dynamic range without log distortion.
    """

    name = "sqrt"

    def get_transform(self) -> Transform:
        """Return the square root transformation object."""
        return _SqrtTransform()

    def set_default_locators_and_formatters(self, axis: Axis) -> None:
        """Configure tick locators and formatters for sqrt scale."""
        axis.set_major_locator(MaxNLocator(5))
        axis.set_major_formatter(FuncFormatter(_sqrt_formatter))

    def limit_range_for_scale(
        self, vmin: float, vmax: float, minpos: float
    ) -> Tuple[float, float]:
        """Clip negative input values and return valid range for sqrt."""
        return max(vmin, 0.0), vmax


class _SqrtTransform(Transform):
    @property
    def output_dims(self) -> int:
        return 1

    @property
    def input_dims(self) -> int:
        return 1

    @property
    def is_separable(self) -> bool:
        return True

    def transform_non_affine(self, values: np.ndarray) -> np.ndarray:  # type: ignore
        """Apply sqrt transform (non-affine)."""
        return np.sqrt(np.clip(values, 0, None))

    def inverted(self) -> Transform:
        """Return the inverse transformation (square)."""
        return _InvertedSqrtTransform()


class _InvertedSqrtTransform(Transform):
    @property
    def output_dims(self) -> int:
        return 1

    @property
    def input_dims(self) -> int:
        return 1

    @property
    def is_separable(self) -> bool:
        return True

    def transform_non_affine(self, values: np.ndarray) -> np.ndarray:  # type: ignore
        """Apply inverse sqrt transform (square)."""
        return values**2

    def inverted(self) -> Transform:
        """Return the forward sqrt transform."""
        return _SqrtTransform()


def _sqrt_formatter(val: float, _: int) -> str:
    """Format major ticks as sqrt-scaled labels."""
    return f"{np.sqrt(val):.1f}" if val >= 0 else ""


def register_sqrt_scale():
    register_scale(SqrtScale)


if __name__ == "__main__":
    register_sqrt_scale()

    x = np.linspace(0, 1, 100)
    y = x**2

    fig, ax = plt.subplots()
    ax.plot(x, y, label=r"$y = x^2$")
    ax.set_yscale("sqrt")  # type: ignore
    ax.legend()
    ax.grid(True)
    ax.set_title("Square Root Y-Scale")
    plt.tight_layout()
    plt.show()
