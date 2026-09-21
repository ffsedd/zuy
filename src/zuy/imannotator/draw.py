import argparse
from pathlib import Path
from typing import Callable, Iterator

from PIL import Image, ImageDraw

from .models import Shape
from .storage import AnnotationStore

# =========================
# CONFIG
# =========================
DEFAULT_SHAPE_SIZE = 80  # keep in sync with the editor (ideally move to models.py)
DEFAULT_SUFFIX = "_x"


# =========================
# DRAWING
# =========================
def _radius(s: Shape) -> float:
    return s.size or DEFAULT_SHAPE_SIZE


def _draw_circle(draw: ImageDraw.ImageDraw, s: Shape) -> None:
    r = _radius(s)
    # contrast stack: white base + red detail
    draw.ellipse([s.x - r, s.y - r, s.x + r, s.y + r], outline="white", width=9)
    draw.ellipse(
        [s.x - r + 3, s.y - r + 3, s.x + r - 3, s.y + r - 3], outline="red", width=3
    )


def _draw_cross(draw: ImageDraw.ImageDraw, s: Shape) -> None:
    r = _radius(s)
    draw.line([s.x - r, s.y, s.x + r, s.y], fill="green", width=2)
    draw.line([s.x, s.y - r, s.x, s.y + r], fill="green", width=2)


def _draw_rectangle(draw: ImageDraw.ImageDraw, s: Shape) -> None:
    if s.x2 is None or s.y2 is None:
        return
    draw.rectangle([s.x, s.y, s.x2, s.y2], outline="blue", width=2)


DRAWERS: dict[str, Callable[[ImageDraw.ImageDraw, Shape], None]] = {
    "circle": _draw_circle,
    "cross": _draw_cross,
    "rectangle": _draw_rectangle,
}


def draw_shape(draw: ImageDraw.ImageDraw, s: Shape) -> None:
    fn = DRAWERS.get(s.type)
    if fn is None:
        print(f"[WARN] unknown shape type: {s.type!r}")
        return
    fn(draw, s)


# =========================
# PATHS
# =========================
def resolve_image(key: str, base: Path) -> Path | None:
    """Keys are relative to the YAML file; legacy keys may be cwd-relative or absolute."""
    candidates = [base / key, Path(key)]
    return next((p for p in candidates if p.is_file()), None)


def out_path_for(img_path: Path, suffix: str, out_dir: Path | None) -> Path:
    name = f"{img_path.stem}{suffix}{img_path.suffix}"
    return (out_dir or img_path.parent) / name


# =========================
# RENDER
# =========================
def render_image(img_path: Path, shapes: list[Shape], out_path: Path) -> None:
    with Image.open(img_path) as im:
        img = im.convert("RGB")
    draw = ImageDraw.Draw(img)
    for s in shapes:
        draw_shape(draw, s)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def iter_entries(store: AnnotationStore) -> Iterator[tuple[str, list[Shape]]]:
    for key, entry in store.data.get("images", {}).items():
        raw_shapes = (entry or {}).get("shapes") or []
        yield key, [Shape.from_dict(s) for s in raw_shapes]


def render(yaml_path: Path, suffix: str = DEFAULT_SUFFIX, out_dir: Path | None = None) -> int:
    store = AnnotationStore(yaml_path)
    store.load_yaml()
    base = yaml_path.parent

    ok = missing = failed = 0
    for key, shapes in iter_entries(store):
        img_path = resolve_image(key, base)
        if img_path is None:
            print(f"[MISS] {key}")
            missing += 1
            continue

        out_path = out_path_for(img_path, suffix, out_dir)
        try:
            render_image(img_path, shapes, out_path)
        except Exception as e:  # keep going on a bad image
            print(f"[FAIL] {img_path}: {e}")
            failed += 1
            continue

        print(f"[OK] {out_path}")
        ok += 1

    print(f"done: {ok} rendered, {missing} missing, {failed} failed")
    return 1 if failed else 0


# =========================
# ENTRY
# =========================
def main() -> int:
    ap = argparse.ArgumentParser(description="Render annotations onto images.")
    ap.add_argument("yaml", nargs="?", default="annotations.yaml", type=Path)
    ap.add_argument("--suffix", default=DEFAULT_SUFFIX)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    return render(args.yaml, args.suffix, args.out_dir)


if __name__ == "__main__":
    raise SystemExit(main())
