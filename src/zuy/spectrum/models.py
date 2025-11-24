from dataclasses import dataclass, field
from typing import Any
import numpy as np


@dataclass
class Spectrum:
    x: np.ndarray
    y: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.x.shape != self.y.shape:
            raise ValueError(
                f"x and y must have same shape, got {self.x.shape} and {self.y.shape}"
            )
