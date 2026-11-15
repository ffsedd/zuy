import pytest
from PIL import Image, ImageChops, ImageDraw

from zuy.imannotator.callouts import (
    callout_distance,
    callout_overlay,
    constrain_callout,
    layout_callout,
)
from zuy.imannotator.draw import draw_shape, render_image
from zuy.imannotator.models import Shape


def label(**kw):
    return Shape(**(dict(type="callout", x=20, y=30, x2=160, y2=100, number=1, font_size=24) | kw))


@pytest.mark.parametrize(
    "tip",
    [(10, 10), (160, 10), (300, 10), (10, 100), (300, 100), (10, 200), (160, 200), (300, 200)],
)
def test_arrow_geometry(tip):
    s = label(x=tip[0], y=tip[1])
    layout = layout_callout(s)
    left, t, r, b = layout.box
    ax, ay = layout.start
    assert left <= ax <= r and t <= ay <= b
    assert min(abs(ax - left), abs(ax - r), abs(ay - t), abs(ay - b)) < 1e-8
    assert layout.head[0] == tip
    assert callout_distance(s, *tip) == 0
    assert callout_distance(s, 160, 100) == 0
    assert callout_distance(s, 1000, 1000) > 0


@pytest.mark.parametrize("size", [12, 24, 60])
def test_unicode_and_edge_constraints(size):
    s = constrain_callout(
        label(x2=0, y2=240, number=None, text="částice α", font_size=size), (400, 240)
    )
    left, t, r, b = layout_callout(s).box
    assert 0 <= left < r <= 400
    assert 0 <= t < b <= 240
    overlay = callout_overlay((400, 240), [s])
    assert overlay.getbbox() is not None


def test_coincident_and_oversize():
    s = label(x=160, y=100)
    assert layout_callout(s).head == ()
    assert callout_overlay((320, 240), [s]).getbbox()
    big = constrain_callout(label(text="long label", number=None, font_size=100), (10, 10))
    assert (big.x2, big.y2) == (5, 5)


def test_export_matches_shared_renderer(tmp_path):
    image = tmp_path / "source.png"
    Image.new("RGB", (320, 240), "gray").save(image)
    output = tmp_path / "export.png"
    shapes = [Shape("cross", 40, 50, size=10), label()]
    render_image(image, shapes, output)
    expected = Image.open(image).convert("RGB")
    draw_shape(ImageDraw.Draw(expected), shapes[0])
    expected.paste(
        callout_overlay(expected.size, shapes), (0, 0), callout_overlay(expected.size, shapes)
    )
    with Image.open(output) as actual:
        assert ImageChops.difference(actual, expected).getbbox() is None


def test_color_used_in_preview_and_export(tmp_path):
    from dataclasses import replace

    color = (224, 32, 64)
    shape = label(color="#e02040")
    overlay = callout_overlay((320, 240), [shape])
    assert (*color, 255) in overlay.getdata()
    assert (0, 0, 0, 255) not in overlay.getdata()
    preview = callout_overlay((160, 120), [shape], scale=0.5)
    assert (*color, 255) in preview.getdata()
    source = tmp_path / "source.png"
    output = tmp_path / "colored.png"
    Image.new("RGB", (320, 240), "gray").save(source)
    render_image(source, [shape], output)
    with Image.open(output) as exported:
        assert color in exported.getdata()
    # Missing color in old YAML preserves the old black appearance.
    legacy = callout_overlay((320, 240), [label()])
    explicit_black = callout_overlay((320, 240), [replace(shape, color="#000000")])
    assert legacy.tobytes() == explicit_black.tobytes()


@pytest.mark.parametrize("color", ["red", "#fff", "#gg0011", "", 123])
def test_invalid_callout_color(color):
    with pytest.raises(ValueError, match="color"):
        label(color=color)
