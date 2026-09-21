from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any, Literal

ShapeType = Literal["circle", "cross", "rectangle"]


@dataclass(frozen=True)
class Shape:
    type: ShapeType
    x: float
    y: float
    size: float | None = None
    x2: float | None = None
    y2: float | None = None

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
