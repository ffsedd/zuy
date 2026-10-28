"""Exercise actual editor handlers with Tcl variables and an in-memory canvas."""

import tkinter as tk
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from zuy.imannotator.gui import ImageEditor, Viewport
from zuy.imannotator.models import Shape
from zuy.imannotator.storage import AnnotationStore


@pytest.fixture
def editor(tmp_path, monkeypatch):
    interp = tk.Tcl()
    e = ImageEditor.__new__(ImageEditor)
    e.root = Mock()
    e.canvas = Mock()
    e.renderer = Mock()
    e.tool = "numbered"
    e.size = tk.IntVar(interp, value=80)
    e.status = tk.StringVar(interp)
    e.label_number = tk.StringVar(interp, value="1")
    e.label_text = tk.StringVar(interp, value="částice α")
    e.font_size = tk.StringVar(interp, value="24")
    e.tool_var = tk.StringVar(interp, value=e.tool)
    e.view = Viewport()
    e.img = Image.new("RGB", (400, 300), "gray")
    e.photo = None
    e.shapes = []
    e.undo_stack = []
    e.redo_stack = []
    e.dirty = False
    e.reviewed = 0
    e.drag_start = None
    e.hover = None
    e._number_shapes = None
    e._overlay_key = None
    e._overlay_photo = None
    e._preview_photo = None
    e.files = [tmp_path / "first.png", tmp_path / "second.png"]
    for path in e.files:
        e.img.save(path)
    e.i = 0
    e.store = AnnotationStore(tmp_path / "annotations.yaml")
    monkeypatch.setattr("zuy.imannotator.gui.ImageTk.PhotoImage", lambda image: image.copy())
    monkeypatch.setattr("zuy.imannotator.gui.messagebox.showerror", Mock())
    e.redraw()
    return e


def event(x, y):
    return SimpleNamespace(x=x, y=y)


def place(e, tip=(30, 40), label=(200, 40)):
    e.on_click(event(*tip))
    e.on_drag(event(*label))
    e.on_release(event(*label))


def test_placement_preview_numbering_and_history(editor):
    e = editor
    e.on_click(event(30, 40))
    e.on_drag(event(200, 40))
    assert e._preview_photo is not None
    assert not e.shapes and not e.undo_stack
    e.on_release(event(200, 40))
    assert e.shapes[0].number == 1
    assert e.label_number.get() == "2"
    e.label_number.set("7")
    place(e, label=(30, 200))  # vertical is valid too
    assert e.shapes[-1].number == 7
    assert e.label_number.get() == "8"
    e.undo()
    assert e.label_number.get() == "2"
    e.redo()
    assert e.label_number.get() == "8"
    e.delete_last()
    assert e.label_number.get() == "2"
    e.clear()
    assert not e.shapes and e.label_number.get() == "1"


def test_invalid_placements_do_not_change_history(editor):
    e = editor
    e.on_click(event(450, 30))
    assert e.drag_start is None
    place(e, label=(31, 41))
    assert not e.shapes and not e.undo_stack
    place(e)
    history = len(e.undo_stack)
    e.label_number.set("1")
    place(e)
    assert len(e.shapes) == 1 and len(e.undo_stack) == history
    e.label_number.set("oops")
    place(e)
    assert len(e.shapes) == 1
    e.set_tool("text")
    e.label_text.set(" ")
    place(e)
    assert len(e.shapes) == 1
    e.label_text.set("bad\ntext")
    place(e)
    assert len(e.shapes) == 1


def test_text_cancellation_and_cache(editor):
    e = editor
    e.set_tool("text")
    e.on_click(event(30, 40))
    e.on_drag(event(200, 100))
    e.cancel_drag()
    e.on_release(event(200, 100))
    assert not e.shapes and not e.undo_stack
    place(e)
    assert e.shapes[0].text == "částice α"
    assert e.label_text.get() == "částice α"
    cached = e._overlay_photo
    e.redraw()
    assert e._overlay_photo is cached
    e.view.scale = 0.5
    e.redraw()
    assert e._overlay_photo.size == (200, 150)
    e.on_click(event(10, 10))
    e.set_tool("numbered")
    assert e.drag_start is None


def test_save_reload_navigation_and_delete_hit(editor):
    e = editor
    place(e)
    e.shapes.append(Shape("cross", 180, 40, size=10))
    e.on_delete_nearest(event(200, 40))
    assert [s.type for s in e.shapes] == ["cross"]
    e.undo()
    e.save()
    saved = list(e.shapes)
    e.update_view = Mock()
    e.on_click(event(20, 20))
    e.next()
    assert e.drag_start is None and not e.shapes
    e.prev()
    assert e.shapes == saved
    assert not e.undo_stack and not e.redo_stack
    e.redraw()
    assert e.label_number.get() == "2"


@pytest.mark.parametrize("widget_type", [tk.Entry, tk.Spinbox, tk.Text])
def test_shortcuts_ignored_in_inputs(editor, widget_type):
    callback = Mock()
    widget = widget_type.__new__(widget_type)
    editor._key(SimpleNamespace(widget=widget), callback)
    callback.assert_not_called()
    editor._key(SimpleNamespace(widget=editor.canvas), callback)
    callback.assert_called_once()


def test_gui_save_reload_export_workflow(editor, tmp_path):
    from PIL import ImageChops, ImageDraw

    from zuy.imannotator.callouts import callout_overlay
    from zuy.imannotator.draw import draw_shape, render

    e = editor
    place(e)
    e.set_tool("text")
    place(e, tip=(80, 250), label=(240, 230))
    # A later legacy mark must remain below callouts, like the canvas overlay.
    e.shapes.append(Shape("cross", 200, 40, size=30))
    e.save()
    expected_shapes = list(e.shapes)
    e.store = AnnotationStore(e.store.path)
    e.store.load_yaml()
    e.update_view = Mock()
    e.load()
    assert e.shapes == expected_shapes
    assert e.store.data["version"] == 2
    output = tmp_path / "exported"
    assert render(e.store.path, out_dir=output) == 0
    expected = e.img.copy()
    draw = ImageDraw.Draw(expected)
    for shape in e.shapes:
        if shape.type != "callout":
            draw_shape(draw, shape)
    overlay = callout_overlay(e.img.size, e.shapes)
    expected.paste(overlay, (0, 0), overlay)
    with Image.open(output / "first_x.png") as exported:
        assert ImageChops.difference(exported, expected).getbbox() is None
    for scale in (0.5, 1.0, 1.5):
        e.view.scale = scale
        e.redraw()
        resized = overlay.resize(
            (round(400 * scale), round(300 * scale)), Image.Resampling.LANCZOS
        )
        assert e._overlay_photo.tobytes() == resized.tobytes()
