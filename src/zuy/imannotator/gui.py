import sys
import tkinter as tk
from pathlib import Path
from typing import List

from PIL import Image, ImageTk

from .models import Shape
from .storage import AnnotationStore

# =========================
# CONFIG
# =========================
OUT_SUFFIX = "_x"

DEFAULT_SHAPE_SIZE = 80


# =========================
# COORD HELPERS
# =========================
def to_img(x, y, scale):
    return x / scale, y / scale


def to_screen(x, y, scale):
    return x * scale, y * scale


# =========================
# EDITOR
# =========================
class ImageEditor:
    def __init__(self, root: tk.Tk, files: List[Path], log_path: Path):
        self.root = root
        self.files = files
        self.i = 0

        self.store = AnnotationStore(log_path)
        self.store.load_yaml()

        self.tool = "circle"
        self.size = tk.IntVar(value=DEFAULT_SHAPE_SIZE)

        self.scale = 1.0
        self.img: Image.Image | None = None
        self.view = None
        self.tk = None

        self.shapes: List[Shape] = []
        self.start = None

        self._ui()
        self._bind()
        self.load()

    # ---------------- UI ----------------
    def _ui(self):
        self.root.title("YAML Image Annotator")

        bar = tk.Frame(self.root)
        bar.pack(fill=tk.X)

        tk.Button(bar, text="Prev (P)", command=self.prev).pack(side=tk.LEFT)
        tk.Button(bar, text="Next (N)", command=self.next).pack(side=tk.LEFT)

        tk.Button(bar, text="Circle (C)", command=lambda: self.set_tool("circle")).pack(
            side=tk.LEFT
        )
        tk.Button(bar, text="Cross (X)", command=lambda: self.set_tool("cross")).pack(side=tk.LEFT)
        tk.Button(bar, text="Rect (R)", command=lambda: self.set_tool("rectangle")).pack(
            side=tk.LEFT
        )

        tk.Label(bar, text="Size").pack(side=tk.LEFT)
        tk.Spinbox(bar, from_=1, to=300, textvariable=self.size, width=6).pack(side=tk.LEFT)

        tk.Button(bar, text="Save (S)", command=self.save).pack(side=tk.RIGHT)

        self.canvas = tk.Canvas(self.root, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

    # ---------------- BINDINGS ----------------
    def _bind(self):
        self.canvas.bind("<Button-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.root.bind("<Key-s>", lambda e: self.save())
        self.root.bind("<Key-n>", lambda e: self.next())
        self.root.bind("<Key-p>", lambda e: self.prev())

        self.root.bind("<Key-c>", lambda e: self.set_tool("circle"))
        self.root.bind("<Key-x>", lambda e: self.set_tool("cross"))
        self.root.bind("<Key-r>", lambda e: self.set_tool("rectangle"))

    # ---------------- LOAD ----------------
    def load(self):
        self.shapes = []

        path = str(self.files[self.i])

        self.shapes = self.store.get_shapes(str(self.files[self.i]))

        self.img = Image.open(path).convert("RGB")
        self.update_view()

    def update_view(self):
        w, h = self.img.size  # type: ignore

        self.root.update_idletasks()
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        self.scale = min(cw / w, ch / h)

        nw, nh = int(w * self.scale), int(h * self.scale)

        self.view = self.img.resize((nw, nh), Image.Resampling.LANCZOS)  # type: ignore
        self.tk = ImageTk.PhotoImage(self.view)

        self.canvas.config(width=nw, height=nh)
        self.redraw()

    # ---------------- RENDER ----------------
    def redraw(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk)

        for s in self.shapes:
            self.draw_shape(s)

    def draw_shape(self, s: Shape):
        if s.type == "circle":
            x, y = to_screen(s.x, s.y, self.scale)
            r = (s.size or DEFAULT_SHAPE_SIZE) * self.scale

            # white halo + red core
            self.canvas.create_oval(x - r, y - r, x + r, y + r, outline="white", width=3)
            self.canvas.create_oval(x - r, y - r, x + r, y + r, outline="red", width=1)

        elif s.type == "cross":
            x, y = to_screen(s.x, s.y, self.scale)
            r = (s.size or DEFAULT_SHAPE_SIZE) * self.scale

            self.canvas.create_line(x - r, y, x + r, y, fill="green", width=2)
            self.canvas.create_line(x, y - r, x, y + r, fill="green", width=2)

        elif s.type == "rectangle":
            x1, y1 = to_screen(s.x, s.y, self.scale)
            x2, y2 = to_screen(s.x2, s.y2, self.scale)

            self.canvas.create_rectangle(x1, y1, x2, y2, outline="blue", width=2)

    # ---------------- INPUT ----------------
    def on_click(self, e):
        ix, iy = to_img(e.x, e.y, self.scale)
        size = self.size.get()

        if self.tool in ("circle", "cross"):
            self.shapes.append(Shape(self.tool, ix, iy, size=size))
            self.redraw()

        elif self.tool == "rectangle":
            self.start = (ix, iy)

    def on_drag(self, e):
        if self.tool != "rectangle" or not self.start:
            return

        self.redraw()

        x0, y0 = self.start
        sx, sy = to_screen(x0, y0, self.scale)

        self.canvas.create_rectangle(sx, sy, e.x, e.y, outline="blue")

    def on_release(self, e):
        if self.tool != "rectangle" or not self.start:
            return

        ix, iy = to_img(e.x, e.y, self.scale)
        x0, y0 = self.start

        self.shapes.append(Shape("rectangle", x0, y0, x2=ix, y2=iy))
        self.start = None
        self.redraw()

    # ---------------- NAV ----------------
    def next(self):
        self.save()
        self.i = min(self.i + 1, len(self.files) - 1)
        self.load()

    def prev(self):
        self.save()
        self.i = max(self.i - 1, 0)
        self.load()

    def set_tool(self, t):
        self.tool = t

    # ---------------- SAVE ----------------
    def save(self):
        self.store.add_image(str(self.files[self.i]), self.shapes)
        self.store.save_yaml()


# =========================
# RUN
# =========================
def main():
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    files = sorted(folder.glob("*.jpg"))

    if not files:
        print("No images found")
        return

    root = tk.Tk()
    root.geometry("1400x900")

    log_path = folder / "annotations.yaml"

    ImageEditor(root, files, log_path)
    root.mainloop()


if __name__ == "__main__":
    main()
