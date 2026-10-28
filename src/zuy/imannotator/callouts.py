"""Image-space callout layout shared by the editor and exported images."""

from dataclasses import dataclass, replace
from functools import lru_cache
from math import hypot

from PIL import Image, ImageDraw, ImageFont

from .models import Shape


@lru_cache(maxsize=64)
def callout_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Prefer DejaVu Sans; Pillow's built-in font is the explicit fallback."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


@dataclass(frozen=True)
class CalloutLayout:
    box: tuple[float, float, float, float]
    text_origin: tuple[float, float]
    start: tuple[float, float]
    tip: tuple[float, float]
    head: tuple[tuple[float, float], ...]
    width: int


def layout_callout(s: Shape) -> CalloutLayout:
    if s.type != "callout" or s.x2 is None or s.y2 is None or s.font_size is None:
        raise ValueError("Expected a validated callout")
    font = callout_font(max(1, round(s.font_size)))
    text = s.text if s.text is not None else str(s.number)
    left, top, right, bottom = font.getbbox(text)
    padding = max(2.0, s.font_size * 0.25)
    w, h = right - left + 2 * padding, bottom - top + 2 * padding
    if s.number is not None:
        w = h = max(w, h)
    cx, cy = s.x2, s.y2
    box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
    dx, dy = s.x - cx, s.y - cy
    ratio = max(abs(dx) / (w / 2), abs(dy) / (h / 2))
    start = (cx + dx / ratio, cy + dy / ratio) if ratio > 1 else (s.x, s.y)
    length = hypot(s.x - start[0], s.y - start[1])
    head: tuple[tuple[float, float], ...] = ()
    if length:
        ux, uy = (s.x - start[0]) / length, (s.y - start[1]) / length
        depth = min(length, s.font_size * 0.6)
        half = depth * 0.45
        bx, by = s.x - ux * depth, s.y - uy * depth
        head = ((s.x, s.y), (bx - uy * half, by + ux * half), (bx + uy * half, by - ux * half))
    return CalloutLayout(
        box,
        (cx - (right - left) / 2 - left, cy - (bottom - top) / 2 - top),
        start,
        (s.x, s.y),
        head,
        max(1, round(s.font_size / 12)),
    )


def constrain_callout(s: Shape, size: tuple[int, int]) -> Shape:
    """Keep the label box inside the image when it fits; center oversize boxes."""
    box = layout_callout(s).box
    assert s.x2 is not None and s.y2 is not None
    half_w, half_h = (box[2] - box[0]) / 2, (box[3] - box[1]) / 2
    w, h = size
    return replace(
        s,
        x2=max(half_w, min(w - half_w, s.x2)) if 2 * half_w <= w else w / 2,
        y2=max(half_h, min(h - half_h, s.y2)) if 2 * half_h <= h else h / 2,
    )


def draw_callout(draw: ImageDraw.ImageDraw, s: Shape) -> None:
    layout = layout_callout(s)
    if layout.head:
        draw.line([layout.start, layout.tip], fill="white", width=layout.width * 3)
        draw.polygon(layout.head, fill="black")
        draw.line([layout.start, layout.tip], fill="black", width=layout.width)
    draw.rounded_rectangle(
        layout.box,
        radius=(layout.box[3] - layout.box[1]) / 4,
        fill="white",
        outline="black",
        width=layout.width,
    )
    draw.text(
        layout.text_origin,
        s.text if s.text is not None else str(s.number),
        fill="black",
        font=callout_font(max(1, round(s.font_size or 1))),
    )


def callout_overlay(size: tuple[int, int], shapes: list[Shape]) -> Image.Image:
    overlay = Image.new("RGBA", size)
    draw = ImageDraw.Draw(overlay)
    for shape in shapes:
        if shape.type == "callout":
            draw_callout(draw, shape)
    return overlay


def callout_distance(s: Shape, x: float, y: float) -> float:
    """Squared distance to the label box or visible leader segment."""
    layout = layout_callout(s)
    left, t, r, b = layout.box
    box_distance = max(left - x, 0, x - r) ** 2 + max(t - y, 0, y - b) ** 2
    ax, ay = layout.start
    bx, by = layout.tip
    dx, dy = bx - ax, by - ay
    length_squared = dx * dx + dy * dy
    u = max(0, min(1, ((x - ax) * dx + (y - ay) * dy) / length_squared)) if length_squared else 0
    return min(box_distance, (x - ax - u * dx) ** 2 + (y - ay - u * dy) ** 2)


def next_number(shapes: list[Shape]) -> int:
    return (
        max((s.number for s in shapes if s.type == "callout" and s.number is not None), default=0)
        + 1
    )
