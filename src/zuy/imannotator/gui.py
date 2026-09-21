import sys
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional

from PIL import Image, ImageTk

from .models import Shape, ShapeType
from .storage import AnnotationStore

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


TOOLS: dict[ShapeType, ToolStyle] = {
    "circle": ToolStyle("Circle", "c"),
    "cross": ToolStyle("Cross", "x"),
    "rectangle": ToolStyle("Rect", "r"),
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

        self.tool: ShapeType = "circle"
        self.size = tk.IntVar(value=DEFAULT_SHAPE_SIZE)
        self.status = tk.StringVar()

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

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _key(self, e, fn):
        if isinstance(e.widget, tk.Spinbox):
            return
        fn()

    # ---------------- LOAD / VIEW ----------------
    @property
    def path(self) -> str:
        return str(self.files[self.i])

    def load(self):
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
        for s in self.shapes:
            self.renderer.draw(s)
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
        if self.tool in ("circle", "cross"):
            self._push_undo()
            self.shapes.append(Shape(self.tool, ix, iy, size=self.size.get()))
            self.redraw()
        else:
            self.drag_start = (ix, iy)

    def on_drag(self, e):
        if self.tool != "rectangle" or not self.drag_start:
            return
        self.redraw()
        ix, iy = self.view.to_img(e.x, e.y)
        x0, y0 = self.drag_start
        self.renderer.draw(Shape("rectangle", x0, y0, x2=ix, y2=iy), preview=True)

    def on_release(self, e):
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
            d = (cx - ix) ** 2 + (cy - iy) ** 2
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
def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)

    if not files:
        print("No images found")
        return

    root = tk.Tk()
    root.geometry("1400x900")
    ImageEditor(root, files, folder / "annotations.yaml")
    root.mainloop()


if __name__ == "__main__":
    main()
