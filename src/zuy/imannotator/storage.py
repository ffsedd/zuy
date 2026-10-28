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

        version = raw.get("version", 1)
        if type(version) is not int or version not in (1, 2):
            raise ValueError(f"Unsupported annotation version: {version!r}")
        for image, entry in (raw.get("images") or {}).items():
            self._parse_shapes(image, (entry or {}).get("shapes") or [])
        return {
            "version": version,
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
        self._parse_shapes(image, new_shapes)
        if any(s.type == "callout" for s in shapes):
            self.data["version"] = 2

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
        return self._parse_shapes(image, entry.get("shapes") or [])

    @staticmethod
    def _parse_shapes(image: str, raw_shapes: list[Any]) -> list[Shape]:
        shapes = []
        numbers: set[int] = set()
        for index, raw in enumerate(raw_shapes):
            try:
                shape = Shape.from_dict(raw)
                if shape.type == "callout" and shape.number is not None:
                    if shape.number in numbers:
                        raise ValueError(f"duplicate callout number {shape.number}")
                    numbers.add(shape.number)
                shapes.append(shape)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Image {image!r}, shape {index + 1}: {exc}") from exc
        return shapes

    def is_annotated(self, image: str) -> bool:
        """True if the image has been saved at all, even with zero shapes."""
        return self._find_entry(image) is not None
