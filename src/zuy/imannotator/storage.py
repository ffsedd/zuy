from __future__ import annotations

import os
import tempfile
from datetime import datetime, timezone
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

    def _normalize(self, raw: Any) -> dict[str, Any]:
        if not isinstance(raw, dict):
            return {"version": 1, "images": {}}

        return {
            "version": raw.get("version", 1),
            "images": raw.get("images", {}) or {},
        }

    # -------------------------
    # WRITE (atomic)
    # -------------------------
    def save_yaml(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=self.path.parent, prefix=self.path.name, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                yaml.safe_dump(
                    self.data,
                    f,
                    sort_keys=False,
                    allow_unicode=True,
                    default_flow_style=False,
                )
            os.replace(tmp, self.path)  # atomic on same filesystem
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    # -------------------------
    # KEYS
    # -------------------------
    def _key(self, image: str) -> str:
        """Key images relative to the YAML file so the folder can be moved."""
        try:
            rel = Path(image).resolve().relative_to(self.path.parent.resolve())
            return rel.as_posix()
        except ValueError:
            return image

    def _find_entry(self, image: str) -> dict[str, Any] | None:
        images = self.data["images"]
        # new-style key first, then legacy key exactly as older versions stored it
        return images.get(self._key(image)) or images.get(image)

    # -------------------------
    # API
    # -------------------------
    def add_image(self, image: str, shapes: list[Shape]) -> None:
        new_shapes = [s.to_dict() for s in shapes]

        existing = self._find_entry(image)
        if existing and existing.get("shapes") == new_shapes:
            return  # unchanged, keep the old timestamp

        key = self._key(image)
        self.data["images"].pop(image, None)  # migrate legacy key
        self.data["images"][key] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "shapes": new_shapes,
        }

    def get_shapes(self, image: str) -> list[Shape]:
        entry = self._find_entry(image)
        if not entry:
            return []
        return [Shape.from_dict(s) for s in entry.get("shapes") or []]

    def is_annotated(self, image: str) -> bool:
        """True if the image has been saved at all, even with zero shapes."""
        return self._find_entry(image) is not None
