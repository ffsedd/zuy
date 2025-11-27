from dataclasses import dataclass, field

import numpy as np


@dataclass
class Spectrum:
    x: np.ndarray
    y: np.ndarray
    metadata: dict[str, float | str] = field(default_factory=dict)

    def __post_init__(self):
        if self.x.shape != self.y.shape:
            raise ValueError(
                f"x and y must have same shape, got {self.x.shape} and {self.y.shape}"
            )
        if not np.all(np.diff(self.x) > 0):
            raise ValueError("x values must be strictly increasing")

    def copy(self) -> "Spectrum":
        return Spectrum(x=self.x.copy(), y=self.y.copy(), metadata=self.metadata.copy())

    def __len__(self) -> int:
        return len(self.x)

    def __repr__(self) -> str:
        return f"Spectrum(x=[{len(self.x)} pts], y=[{len(self.y)} pts], metadata={self.metadata})"
