from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from math import isfinite
from typing import Any, Literal

ShapeType = Literal["circle", "cross", "rectangle", "callout"]


@dataclass(frozen=True)
class Shape:
    type: ShapeType
    x: float
    y: float
    size: float | None = None
    x2: float | None = None
    y2: float | None = None

    text: str | None = None
    number: int | None = None
    font_size: float | None = None

    def __post_init__(self) -> None:
        if self.type != "callout":
            return
        for name in ("x", "y", "x2", "y2", "font_size"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(value)
            ):
                raise ValueError(f"callout {name} must be a finite number")
        if self.font_size is None or self.font_size <= 0:
            raise ValueError("callout font_size must be positive")
        if (self.text is None) == (self.number is None):
            raise ValueError("callout requires exactly one of text or number")
        if self.text is not None and (
            not isinstance(self.text, str)
            or not self.text.strip()
            or "\n" in self.text
            or "\r" in self.text
        ):
            raise ValueError("callout text must be non-empty single-line text")
        if self.number is not None and (
            isinstance(self.number, bool) or not isinstance(self.number, int) or self.number <= 0
        ):
            raise ValueError("callout number must be a positive integer")

    @property
    def center(self) -> tuple[float, float]:
        if self.type == "rectangle" and self.x2 is not None and self.y2 is not None:
            return (self.x + self.x2) / 2, (self.y + self.y2) / 2
        return self.x, self.y

    def to_dict(self) -> dict[str, Any]:
        """Serialize, omitting unused (None) fields."""
        return {k: v for k, v in asdict(self).items() if v is not None}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Shape:
        """Deserialize, ignoring unknown keys so older/newer files still load."""
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in known})
