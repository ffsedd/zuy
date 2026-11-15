import argparse
import tkinter as tk
from dataclasses import dataclass
from math import hypot, isfinite
from tkinter import colorchooser, messagebox
from pathlib import Path
from typing import Callable, List, Optional, Literal

from PIL import Image, ImageTk

from .callouts import callout_distance, callout_overlay, constrain_callout, next_number
from .models import Shape

from .storage import AnnotationStore

ToolType = Literal["circle", "cross", "rectangle", "numbered", "text"]

# =========================
# CONFIG
# =========================
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
DEFAULT_SHAPE_SIZE = 80
MIN_DRAG_PX = 3
UNDO_LIMIT = 100


@dataclass(frozen=True)
class ToolStyle:
    label: str
    key: str


TOOLS: dict[ToolType, ToolStyle] = {
    "circle": ToolStyle("Circle", "c"),
    "cross": ToolStyle("Cross", "x"),
    "rectangle": ToolStyle("Rect", "r"),
    "numbered": ToolStyle("Numbered label", "b"),
    "text": ToolStyle("Text label", "t"),
}


# =========================
# VIEWPORT (image <-> screen)
# =========================
class Viewport:
    def __init__(self):
        self.scale = 1.0

    def fit(self, img_size, canvas_size):
        (w, h), (cw, ch) = img_size, canvas_size
        self.scale = min(max(1, cw) / w, max(1, ch) / h)

    def to_img(self, x, y):
        return x / self.scale, y / self.scale

    def to_screen(self, x, y):
        return x * self.scale, y * self.scale


# =========================
# SHAPE RENDERING
# =========================
class ShapeRenderer:
    """Draws shapes on a canvas. Add a new shape type by adding a _draw_<type>."""

    def __init__(self, canvas: tk.Canvas, view: Viewport):
        self.c = canvas
        self.v = view

    def draw(self, s: Shape, preview: bool = False):
        fn: Optional[Callable] = getattr(self, f"_draw_{s.type}", None)
        if fn:
            fn(s, dash=(4, 4) if preview else None)

    def _radius(self, s):
        return (s.size or DEFAULT_SHAPE_SIZE) * self.v.scale

    def _draw_circle(self, s, dash=None):
        x, y = self.v.to_screen(s.x, s.y)
        r = self._radius(s)
        box = (x - r, y - r, x + r, y + r)
        self.c.create_oval(*box, outline="white", width=3, dash=dash)  # halo
        self.c.create_oval(*box, outline="red", width=1, dash=dash)

    def _draw_cross(self, s, dash=None):
        x, y = self.v.to_screen(s.x, s.y)
        r = self._radius(s)
        self.c.create_line(x - r, y, x + r, y, fill="green", width=2, dash=dash)
        self.c.create_line(x, y - r, x, y + r, fill="green", width=2, dash=dash)

    def _draw_rectangle(self, s, dash=None):
        x1, y1 = self.v.to_screen(s.x, s.y)
        x2, y2 = self.v.to_screen(s.x2, s.y2)
        self.c.create_rectangle(x1, y1, x2, y2, outline="blue", width=2, dash=dash)


# =========================
# EDITOR
# =========================
class ImageEditor:
    def __init__(self, root: tk.Tk, files: List[Path], log_path: Path):
        self.root = root
        self.files = files
        self.i = 0

        self.store = AnnotationStore(log_path)
        if log_path.is_file():
            self.store.load_yaml()

        self.tool: ToolType = "circle"
        self.size = tk.IntVar(value=DEFAULT_SHAPE_SIZE)
        self.status = tk.StringVar()
        self.label_text = tk.StringVar()
        self.label_number = tk.StringVar(value="1")
        self.font_size = tk.StringVar(value="32")
        self.label_color = tk.StringVar(value="#000000")
        self._number_shapes = None
        self._overlay_key = None
        self._overlay_photo = None
        self._preview_photo = None

        self.view = Viewport()
        self.img: Optional[Image.Image] = None
        self.photo: Optional[ImageTk.PhotoImage] = None

        self.shapes: List[Shape] = []
        self.undo_stack: List[List[Shape]] = []
        self.redo_stack: List[List[Shape]] = []
        self.dirty = False
        self.reviewed = 0
        self._count_reviewed()

        self.drag_start = None  # image coords
        self.hover = None  # screen coords

        self._build_ui()
        self.renderer = ShapeRenderer(self.canvas, self.view)
        self._bind()
        self.root.after(50, self.load)  # wait for real canvas size

    # ---------------- UI ----------------
    def _build_ui(self):
        self.root.title("YAML Image Annotator")

        bar = tk.Frame(self.root)
        bar.pack(fill=tk.X)

        tk.Button(bar, text="Prev (P)", command=self.prev).pack(side=tk.LEFT)
        tk.Button(bar, text="Next (N)", command=self.next).pack(side=tk.LEFT)

        self.tool_var = tk.StringVar(value=self.tool)
        for name, t in TOOLS.items():
            tk.Radiobutton(
                bar,
                text=f"{t.label} ({t.key.upper()})",
                variable=self.tool_var,
                value=name,
                indicatoron=False,
                command=lambda n=name: self.set_tool(n),
            ).pack(side=tk.LEFT)

        tk.Label(bar, text="Size").pack(side=tk.LEFT)
        self.spin = tk.Spinbox(bar, from_=1, to=300, textvariable=self.size, width=6)
        self.spin.pack(side=tk.LEFT)

        tk.Button(bar, text="Undo", command=self.undo).pack(side=tk.LEFT)
        tk.Button(bar, text="Redo", command=self.redo).pack(side=tk.LEFT)
        tk.Button(bar, text="Clear", command=self.clear).pack(side=tk.LEFT)
        tk.Button(bar, text="Save (S)", command=self.save).pack(side=tk.RIGHT)

        labels = tk.Frame(self.root)
        labels.pack(fill=tk.X)
        tk.Label(labels, text="Text").pack(side=tk.LEFT)
        tk.Entry(labels, textvariable=self.label_text, width=35).pack(side=tk.LEFT)
        tk.Label(labels, text="Next number").pack(side=tk.LEFT)
        tk.Spinbox(labels, from_=1, to=999999, textvariable=self.label_number, width=7).pack(
            side=tk.LEFT
        )
        tk.Label(labels, text="Font size (image px)").pack(side=tk.LEFT)
        tk.Spinbox(labels, from_=1, to=300, textvariable=self.font_size, width=6).pack(
            side=tk.LEFT
        )
        self.color_button = tk.Button(
            labels, text="Color…", command=self.choose_label_color, bg="#000000", fg="white"
        )
        self.color_button.pack(side=tk.LEFT)
        tk.Label(labels, text="Drag feature → label; Esc cancels").pack(side=tk.LEFT)

        tk.Label(self.root, textvariable=self.status, anchor="w").pack(side=tk.BOTTOM, fill=tk.X)

        self.canvas = tk.Canvas(self.root, bg="black", highlightthickness=0)
        self.canvas.pack(fill=tk.BOTH, expand=True)

    # ---------------- BINDINGS ----------------
    def _bind(self):
        c = self.canvas
        c.bind("<Button-1>", self.on_click)
        c.bind("<B1-Motion>", self.on_drag)
        c.bind("<ButtonRelease-1>", self.on_release)
        c.bind("<Button-3>", self.on_delete_nearest)
        c.bind("<Motion>", self.on_motion)
        c.bind("<Leave>", lambda e: self._set_hover(None))
        c.bind("<MouseWheel>", self.on_wheel)  # Win/macOS
        c.bind("<Button-4>", lambda e: self._bump_size(+5))  # Linux
        c.bind("<Button-5>", lambda e: self._bump_size(-5))
        c.bind("<Configure>", self._on_resize)

        # Key shortcuts, ignored while typing in the Spinbox
        keys = {
            "s": self.save,
            "n": self.next,
            "p": self.prev,
            "<Right>": self.next,
            "<Left>": self.prev,
            "<Delete>": self.delete_last,
            "<Control-z>": self.undo,
            "<Control-y>": self.redo,
            "<Control-Z>": self.redo,
        }
        for name, t in TOOLS.items():
            keys[t.key] = lambda n=name: self.set_tool(n)
        for k, fn in keys.items():
            seq = k if k.startswith("<") else f"<Key-{k}>"
            self.root.bind(seq, lambda e, f=fn: self._key(e, f))

        self.root.bind("<Escape>", lambda e: self.cancel_drag())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _key(self, e, fn):
        if isinstance(e.widget, (tk.Spinbox, tk.Entry, tk.Text)):
            return
        fn()

    # ---------------- LOAD / VIEW ----------------
    @property
    def path(self) -> str:
        return str(self.files[self.i])

    def load(self):
        self.drag_start = None
        self.hover = None
        self._number_shapes = None
        self.shapes = self.store.get_shapes(self.path)  # returns a fresh list
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.dirty = False
        self.img = Image.open(self.path).convert("RGB")
        self.update_view()

    def update_view(self):
        if self.img is None:
            return
        cw, ch = self.canvas.winfo_width(), self.canvas.winfo_height()
        self.view.fit(self.img.size, (cw, ch))
        nw = max(1, int(self.img.width * self.view.scale))
        nh = max(1, int(self.img.height * self.view.scale))
        resized = self.img.resize((nw, nh), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(resized)
        self.redraw()

    def _on_resize(self, _e):
        if getattr(self, "_resize_job", None):
            self.root.after_cancel(self._resize_job)
        self._resize_job = self.root.after(100, self.update_view)  # debounce

    # ---------------- RENDER ----------------
    def redraw(self):
        self.canvas.delete("all")
        if self.photo:
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.photo)
        shape_key = tuple(self.shapes)
        number_key = tuple(
            sorted(s.number for s in self.shapes if s.type == "callout" and s.number is not None)
        )
        if self._number_shapes != number_key:
            self.label_number.set(str(next_number(self.shapes)))
            self._number_shapes = number_key
        for s in self.shapes:
            if s.type != "callout":
                self.renderer.draw(s)
        if self.img is not None:
            key = (shape_key, self.img.size, self.view.scale)
            if key != self._overlay_key:
                overlay = callout_overlay(self.img.size, self.shapes)
                self._overlay_photo = ImageTk.PhotoImage(
                    overlay.resize(
                        (
                            max(1, int(self.img.width * self.view.scale)),
                            max(1, int(self.img.height * self.view.scale)),
                        ),
                        Image.Resampling.LANCZOS,
                    )
                )
                self._overlay_key = key
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self._overlay_photo)
        self._draw_cursor_preview()
        self._update_status()

    def _draw_cursor_preview(self):
        if self.hover and self.tool in ("circle", "cross"):
            ix, iy = self.view.to_img(*self.hover)
            self.renderer.draw(Shape(self.tool, ix, iy, size=self.size.get()), preview=True)

    def _update_status(self):
        mark = "*" if self.dirty else ""
        self.status.set(
            f"{self.i + 1}/{len(self.files)}  {self.files[self.i].name}{mark}  "
            f"shapes: {len(self.shapes)}  tool: {self.tool}  size: {self.size.get()}  "
            f"reviewed: {self.reviewed}/{len(self.files)}"
        )

    def _count_reviewed(self):
        self.reviewed = sum(self.store.is_annotated(str(f)) for f in self.files)

    # ---------------- HISTORY ----------------
    def _push_undo(self):
        self.undo_stack.append(list(self.shapes))
        del self.undo_stack[:-UNDO_LIMIT]
        self.redo_stack.clear()
        self.dirty = True

    def undo(self):
        if self.undo_stack:
            self.redo_stack.append(list(self.shapes))
            self.shapes = self.undo_stack.pop()
            self.dirty = True
            self.redraw()

    def redo(self):
        if self.redo_stack:
            self.undo_stack.append(list(self.shapes))
            self.shapes = self.redo_stack.pop()
            self.dirty = True
            self.redraw()

    def delete_last(self):
        if self.shapes:
            self._push_undo()
            self.shapes.pop()
            self.redraw()

    def clear(self):
        if self.shapes:
            self._push_undo()
            self.shapes = []
            self.redraw()

    # ---------------- INPUT ----------------
    def on_click(self, e):
        self.canvas.focus_set()
        ix, iy = self.view.to_img(e.x, e.y)
        if self.tool in ("numbered", "text") and (
            self.img is None or not (0 <= ix < self.img.width and 0 <= iy < self.img.height)
        ):
            return
        if self.tool in ("circle", "cross"):
            self._push_undo()
            self.shapes.append(Shape(self.tool, ix, iy, size=self.size.get()))
            self.redraw()
        else:
            self.drag_start = (ix, iy)

    def cancel_drag(self):
        self.drag_start = None
        self._preview_photo = None
        self.redraw()

    def choose_label_color(self):
        _, color = colorchooser.askcolor(
            color=self.label_color.get(), title="Arrow label color", parent=self.root
        )
        if color is None:
            return
        self.label_color.set(color)
        red, green, blue = (int(color[i : i + 2], 16) for i in (1, 3, 5))
        foreground = "black" if 299 * red + 587 * green + 114 * blue >= 128000 else "white"
        self.color_button.configure(bg=color, fg=foreground)

    def _make_callout(self, tip, label):
        if self.img is None:
            raise ValueError("No image loaded")
        try:
            size = float(self.font_size.get())
            number = int(self.label_number.get()) if self.tool == "numbered" else None
        except ValueError as exc:
            raise ValueError("Use a positive integer number and a positive font size") from exc
        if not isfinite(size) or not 1 <= size <= 300:
            raise ValueError("Font size must be between 1 and 300 image pixels")
        if number is not None and any(
            s.type == "callout" and s.number == number for s in self.shapes
        ):
            raise ValueError(f"Number {number} is already used in this image")
        shape = Shape(
            "callout",
            *tip,
            x2=label[0],
            y2=label[1],
            text=self.label_text.get().strip() if self.tool == "text" else None,
            number=number,
            font_size=size,
            color=self.label_color.get(),
        )
        try:
            return constrain_callout(shape, self.img.size)
        except (OSError, OverflowError) as exc:
            raise ValueError("Unable to render the selected font size") from exc

    def _preview_callout(self, shape):
        size = (
            max(1, int(self.img.width * self.view.scale)),
            max(1, int(self.img.height * self.view.scale)),
        )
        overlay = callout_overlay(size, [shape], scale=self.view.scale)
        self._preview_photo = ImageTk.PhotoImage(overlay)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self._preview_photo)

    def on_drag(self, e):
        if self.tool in ("numbered", "text") and self.drag_start:
            self.redraw()
            try:
                self._preview_callout(
                    self._make_callout(self.drag_start, self.view.to_img(e.x, e.y))
                )
            except ValueError as exc:
                self.status.set(str(exc))
            return
        if self.tool != "rectangle" or not self.drag_start:
            return
        self.redraw()
        ix, iy = self.view.to_img(e.x, e.y)
        x0, y0 = self.drag_start
        self.renderer.draw(Shape("rectangle", x0, y0, x2=ix, y2=iy), preview=True)

    def on_release(self, e):
        if self.tool in ("numbered", "text") and self.drag_start:
            tip = self.drag_start
            self.drag_start = None
            self._preview_photo = None
            label = self.view.to_img(e.x, e.y)
            if hypot(label[0] - tip[0], label[1] - tip[1]) * self.view.scale < MIN_DRAG_PX:
                self.redraw()
                return
            try:
                shape = self._make_callout(tip, label)
            except ValueError as exc:
                self.redraw()
                messagebox.showerror("Invalid label", str(exc), parent=self.root)
                return
            self._push_undo()
            self.shapes.append(shape)
            self.redraw()
            return
        if self.tool != "rectangle" or not self.drag_start:
            return
        x0, y0 = self.drag_start
        self.drag_start = None
        x1, y1 = self.view.to_img(e.x, e.y)

        sx0, sy0 = self.view.to_screen(x0, y0)
        if abs(e.x - sx0) < MIN_DRAG_PX or abs(e.y - sy0) < MIN_DRAG_PX:
            self.redraw()  # accidental click, ignore
            return

        # normalize so (x, y) is top-left and (x2, y2) bottom-right
        self._push_undo()
        self.shapes.append(
            Shape("rectangle", min(x0, x1), min(y0, y1), x2=max(x0, x1), y2=max(y0, y1))
        )
        self.redraw()

    def on_delete_nearest(self, e):
        ix, iy = self.view.to_img(e.x, e.y)
        best, best_d = None, float("inf")
        for idx, s in enumerate(self.shapes):
            cx, cy = s.center
            d = (
                callout_distance(s, ix, iy)
                if s.type == "callout"
                else (cx - ix) ** 2 + (cy - iy) ** 2
            )
            if d < best_d:
                best, best_d = idx, d
        if best is not None:
            self._push_undo()
            del self.shapes[best]
            self.redraw()

    def on_motion(self, e):
        self._set_hover((e.x, e.y))

    def _set_hover(self, pos):
        self.hover = pos
        if self.drag_start is None:
            self.redraw()

    def on_wheel(self, e):
        self._bump_size(+5 if e.delta > 0 else -5)

    def _bump_size(self, d):
        self.size.set(max(1, min(300, self.size.get() + d)))
        self.redraw()

    def set_tool(self, t):
        self._preview_photo = None
        self.tool = t
        self.tool_var.set(t)
        self.drag_start = None
        self.redraw()

    # ---------------- NAV ----------------
    def _go(self, new_i):
        new_i = max(0, min(new_i, len(self.files) - 1))
        if new_i == self.i:
            return
        self.save()
        self.i = new_i
        self.load()

    def next(self):
        self._go(self.i + 1)

    def prev(self):
        self._go(self.i - 1)

    # ---------------- SAVE ----------------
    def save(self):
        self.store.add_image(self.path, self.shapes)
        self.store.save_yaml()
        self.dirty = False
        self._count_reviewed()
        self._update_status()

    def on_close(self):
        self.save()
        self.root.destroy()


# =========================
# RUN
# =========================
def main() -> None:
    parser = argparse.ArgumentParser(description="YAML Image Annotator")
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("."),
        help="Directory containing images",
    )
    args = parser.parse_args()

    folder = args.dir.expanduser().resolve()
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    if not files:
        print(f"No images found in {folder}")
        return

    root = tk.Tk()
    root.geometry("1400x900")
    ImageEditor(root, files, folder / "annotations.yaml")
    root.mainloop()


if __name__ == "__main__":
    main()
