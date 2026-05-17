# storage.py
from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .models import Shape


class AnnotationStore:
    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, Any] = {"version": 1, "images": {}}

    # -------------------------
    # LOAD
    # -------------------------
    def load_yaml(self) -> None:
        if not self.path.exists():
            raise FileNotFoundError(f"Annotation file not found: {self.path}")

        with self.path.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}

        self.data = self._normalize(raw)

    # -------------------------
    # NORMALIZATION
    # -------------------------
    def _normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {"version": 1, "images": {}}

        return {
            "version": raw.get("version", 1),
            "images": raw.get("images", {}) or {},
        }

    # -------------------------
    # WRITE
    # -------------------------
    def save_yaml(self) -> None:
        with self.path.open("w", encoding="utf-8") as f:
            yaml.safe_dump(
                self.data,
                f,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )

    # -------------------------
    # API
    # -------------------------
    def add_image(self, image: str, shapes: list[Shape]) -> None:
        self.data["images"][image] = {
            "ts": datetime.utcnow().isoformat(),
            "shapes": [asdict(s) for s in shapes],
        }

    def get_shapes(self, image: str) -> list[Shape]:
        entry = self.data["images"].get(image)
        if not entry:
            return []

        return [Shape(**s) for s in entry["shapes"]]
