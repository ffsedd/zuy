# models.py
from dataclasses import dataclass


@dataclass
class Shape:
    type: str
    x: float
    y: float
    size: float | None = None
    x2: float | None = None
    y2: float | None = None
