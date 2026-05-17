from pathlib import Path

from PIL import Image, ImageDraw

from .models import Shape
from .storage import AnnotationStore

# =========================================================
# DRAWING CORE
# =========================================================


def draw_shape(draw: ImageDraw.ImageDraw, s: Shape):
    if s.type == "circle":
        x, y = s.x, s.y
        r = s.size or 10

        # contrast stack: white base + red detail
        draw.ellipse([x - r, y - r, x + r, y + r], outline="white", width=9)  # type: ignore
        draw.ellipse([x - r + 3, y - r + 3, x + r - 3, y + r - 3], outline="red", width=3)  # type: ignore

    elif s.type == "cross":
        r = s.size
        draw.line([s.x - r, s.y, s.x + r, s.y], fill="green", width=2)  # type: ignore
        draw.line([s.x, s.y - r, s.x, s.y + r], fill="green", width=2)  # type: ignore

    elif s.type == "rectangle":
        if s.x2 is None or s.y2 is None:
            return
        draw.rectangle([s.x, s.y, s.x2, s.y2], outline="blue", width=2)


# =========================================================
# RENDER ENGINE
# =========================================================


def render(path: Path, suffix="_x"):
    store = AnnotationStore(path)
    store.load_yaml()

    anns = store.data.get("images", {})

    for img_path_str, ann in anns.items():
        img_path = Path(img_path_str)

        if not img_path.exists():
            print(f"[MISS] {img_path}")
            continue

        img = Image.open(img_path).convert("RGB")
        draw = ImageDraw.Draw(img)

        for s in ann["shapes"]:
            shape = Shape(**s)
            draw_shape(draw, shape)

        out_path = img_path.with_name(f"{img_path.stem}{suffix}{img_path.suffix}")
        img.save(out_path)
        print(f"[OK] {out_path}")


# =========================================================
# ENTRY
# =========================================================


def main():

    import sys

    if len(sys.argv) < 2:
        fp = Path("annotations.yaml")
    else:
        fp = Path(sys.argv[1])
    render(fp)


if __name__ == "__main__":
    main()
