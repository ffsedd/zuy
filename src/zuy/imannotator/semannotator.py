import json
import logging
import sys
import tkinter as tk
from collections import namedtuple
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageDraw, ImageTk

LINE_WIDTH = 2
SUFFIX = "x"

Shape = namedtuple("Shape", ["type", "size", "x", "y", "x2", "y2"])


# ----------------------------
# logger
# ----------------------------
class ActionLogger:
    def __init__(self, path="imgannot.log"):
        self.log = logging.getLogger("imgannot")
        self.log.setLevel(logging.INFO)

        fmt = logging.Formatter("%(message)s")

        # file
        file_handler = logging.FileHandler(path, mode="a", encoding="utf-8")
        file_handler.setFormatter(fmt)

        # stdout
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(fmt)

        # avoid duplicate handlers on reload
        if not self.log.handlers:
            self.log.addHandler(file_handler)
            self.log.addHandler(stream_handler)

    def event(self, **data):
        data["ts"] = datetime.utcnow().isoformat()
        self.log.info(json.dumps(data))


# ----------------------------
# geometry helpers
# ----------------------------
def to_img(x, y, scale):
    return x / scale, y / scale


def to_screen(x, y, scale):
    return x * scale, y * scale


# ----------------------------
# app
# ----------------------------
class ImageEditor:
    def __init__(self, root, fpaths):
        self.root = root
        self.root.title("Annotation Tool")

        self.paths = sorted(fpaths)
        self.i = 0

        self.tool = "circle"
        self.size = tk.IntVar(value=40)

        self.start = None
        self.rect_id = None
        self.objects = []

        self.log = ActionLogger()

        # ---------------- UI ----------------
        self.topbar = tk.Frame(root)
        self.topbar.pack(side=tk.TOP, fill=tk.X)

        self._toolbar()

        self.canvas = tk.Canvas(root, bg="black")
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self._bind()
        self.load()

    # ---------------- UI ----------------
    def _toolbar(self):
        tb = self.topbar

        nav = tk.Frame(tb)
        nav.pack(side=tk.LEFT)

        tk.Button(nav, text="◀ Prev (←)", command=self.prev).pack(side=tk.LEFT)
        tk.Button(nav, text="Next (→) ▶", command=self.next).pack(side=tk.LEFT)

        tools = tk.Frame(tb)
        tools.pack(side=tk.LEFT, padx=10)

        tk.Button(tools, text="Circle (C)", command=lambda: self.set_tool("circle")).pack(
            side=tk.LEFT
        )
        tk.Button(tools, text="Rect (R)", command=lambda: self.set_tool("rectangle")).pack(
            side=tk.LEFT
        )
        tk.Button(tools, text="Cross (X)", command=lambda: self.set_tool("cross")).pack(
            side=tk.LEFT
        )

        sizebox = tk.Frame(tb)
        sizebox.pack(side=tk.LEFT, padx=10)

        tk.Label(sizebox, text="Size").pack(side=tk.LEFT)
        tk.Spinbox(sizebox, from_=2, to=200, textvariable=self.size, width=5).pack(side=tk.LEFT)

        act = tk.Frame(tb)
        act.pack(side=tk.RIGHT)

        tk.Button(act, text="Save (S)", command=self.save).pack(side=tk.LEFT)

    # ---------------- bindings ----------------
    def _bind(self):
        self.root.bind("<c>", lambda e: self.set_tool("circle"))
        self.root.bind("<r>", lambda e: self.set_tool("rectangle"))
        self.root.bind("<x>", lambda e: self.set_tool("cross"))

        self.root.bind("<Left>", lambda e: self.prev())
        self.root.bind("<Right>", lambda e: self.next())
        self.root.bind("<s>", lambda e: self.save())

        self.canvas.bind("<Button-1>", self.click)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", self.release)

        self.canvas.bind("<Button-3>", self.save_next)

    # ---------------- logging helper ----------------
    def log_event(self, event, **payload):
        self.log.event(
            event=event,
            image=str(self.paths[self.i]) if self.paths else None,
            index=self.i,
            tool=self.tool,
            **payload,
        )

    # ---------------- image ----------------
    def load(self):
        self.canvas.delete("all")

        path = self.paths[self.i]
        self.root.title(str(path))

        self.img = Image.open(path)
        self.w, self.h = self.img.size

        self.root.update()
        cw = max(1, self.root.winfo_width())
        ch = max(1, self.root.winfo_height())

        self.scale = min(cw / self.w, ch / self.h)

        nw, nh = int(self.w * self.scale), int(self.h * self.scale)

        self.view = self.img.resize((nw, nh), Image.Resampling.LANCZOS)
        self.tk = ImageTk.PhotoImage(self.view)

        self.canvas.config(width=nw, height=nh)
        self.redraw()

        self.log_event("image_load")

    # ---------------- rendering ----------------
    def redraw(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk)

        for o in self.objects:
            x, y = to_screen(o.x, o.y, self.scale)

            if o.type == "circle":
                s = o.size * self.scale
                self.canvas.create_oval(x - s, y - s, x + s, y + s, outline="red")

            elif o.type == "cross":
                s = o.size * self.scale
                self.canvas.create_line(x - s, y, x + s, y, fill="green")
                self.canvas.create_line(x, y - s, x, y + s, fill="green")

            elif o.type == "rectangle":
                x2, y2 = to_screen(o.x2, o.y2, self.scale)
                self.canvas.create_rectangle(x, y, x2, y2, outline="blue")

    # ---------------- input ----------------
    def click(self, e):
        ix, iy = to_img(e.x, e.y, self.scale)
        size = self.size.get()

        if self.tool == "circle":
            self.objects.append(Shape("circle", size, ix, iy, None, None))
            self.log_event("shape_create", shape="circle", x=ix, y=iy, size=size)
            self.redraw()

        elif self.tool == "cross":
            self.objects.append(Shape("cross", size, ix, iy, None, None))
            self.log_event("shape_create", shape="cross", x=ix, y=iy, size=size)
            self.redraw()

        elif self.tool == "rectangle":
            self.start = (ix, iy)
            self.rect_id = self.canvas.create_rectangle(e.x, e.y, e.x, e.y, outline="blue")

    def drag(self, e):
        if self.tool != "rectangle" or not self.start:
            return

        sx, sy = self.start
        self.canvas.coords(self.rect_id, sx * self.scale, sy * self.scale, e.x, e.y)  # type: ignore

    def release(self, e):
        if self.tool != "rectangle" or not self.start:
            return

        ix, iy = to_img(e.x, e.y, self.scale)

        self.objects.append(Shape("rectangle", None, self.start[0], self.start[1], ix, iy))

        self.log_event(
            "shape_create",
            shape="rectangle",
            x1=self.start[0],
            y1=self.start[1],
            x2=ix,
            y2=iy,
        )

        self.start = None
        self.rect_id = None
        self.redraw()

    # ---------------- navigation ----------------
    def next(self):
        self.log_event("navigate_next")

        if self.i >= len(self.paths) - 1:
            self.log_event("exit", reason="end_of_dataset")
            self.root.destroy()
            return

        self.i += 1
        self.objects = []
        self.load()

    def prev(self):
        self.log_event("navigate_prev")

        if self.i <= 0:
            return

        self.i -= 1
        self.objects = []
        self.load()

    # ---------------- save ----------------
    def save(self):
        path = self.paths[self.i]
        out = path.with_name(f"{path.stem}{SUFFIX}{path.suffix}")

        if not self.objects:
            self.log_event("save_skipped", reason="no_objects")
            return

        img = self.img.copy()
        d = ImageDraw.Draw(img)

        for o in self.objects:
            if o.type == "circle":
                x, y = o.x, o.y
                s = o.size
                d.ellipse([x - s, y - s, x + s, y + s], outline="red", width=LINE_WIDTH)

            elif o.type == "cross":
                x, y = o.x, o.y
                s = o.size
                d.line([x - s, y, x + s, y], fill="green", width=LINE_WIDTH)
                d.line([x, y - s, x, y + s], fill="green", width=LINE_WIDTH)

            elif o.type == "rectangle":
                d.rectangle([o.x, o.y, o.x2, o.y2], outline="blue", width=LINE_WIDTH)

        img.save(out)

        self.log_event("save", output=str(out), objects=len(self.objects))

    def save_next(self, _=None):
        self.log_event("save_next")
        self.save()
        self.next()

    # ---------------- state ----------------
    def set_tool(self, t):
        self.tool = t
        self.root.config(cursor="crosshair")
        self.log_event("tool_change")


def main():
    folder = Path(sys.argv[1]) if len(sys.argv) == 2 else Path(".").resolve()
    files = sorted(folder.glob("*.jpg"))

    root = tk.Tk()
    root.geometry("1600x1200")

    ImageEditor(root, files)
    root.mainloop()


# ---------------- run ----------------
if __name__ == "__main__":
    main()
