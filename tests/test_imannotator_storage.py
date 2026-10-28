from pathlib import Path

import pytest
import yaml

from zuy.imannotator.models import Shape
from zuy.imannotator.storage import AnnotationStore


def callout(**changes):
    return Shape(
        **(dict(type="callout", x=10, y=20, x2=100, y2=120, number=1, font_size=24) | changes)
    )


def test_roundtrip(tmp_path):
    image = str(tmp_path / "image.png")
    store = AnnotationStore(tmp_path / "annotations.yaml")
    shapes = [Shape("circle", 1, 2, size=3), callout(), callout(number=None, text="částice α")]
    store.add_image(image, shapes)
    timestamp = store.data["images"]["image.png"]["ts"]
    store.add_image(image, shapes)
    assert store.data["images"]["image.png"]["ts"] == timestamp
    store.save_yaml()
    loaded = AnnotationStore(store.path)
    loaded.load_yaml()
    assert loaded.get_shapes(image) == shapes
    assert loaded.data["version"] == 2
    loaded.add_image(image, [])
    assert loaded.data["version"] == 2


def test_legacy_fixture():
    store = AnnotationStore(Path(__file__).parent / "fixtures/imannotator.yaml")
    store.load_yaml()
    assert store.get_shapes("legacy.png") == [Shape("cross", 12, 34, size=8)]
    assert len(store.get_shapes("mixed.png")) == 3


@pytest.mark.parametrize(
    "changes",
    [
        dict(x2=None),
        dict(y=float("nan")),
        dict(font_size=0),
        dict(font_size=None),
        dict(number=0),
        dict(number=True),
        dict(text="x"),
        dict(number=None),
        dict(number=None, text="  "),
        dict(number=None, text="a\nb"),
    ],
)
def test_invalid_callouts(changes):
    with pytest.raises(ValueError, match="callout"):
        callout(**changes)


@pytest.mark.parametrize("version", [0, 3, True, "2"])
def test_unsupported_versions(tmp_path, version):
    path = tmp_path / "annotations.yaml"
    path.write_text(yaml.safe_dump(dict(version=version, images={})))
    with pytest.raises(ValueError, match="version"):
        AnnotationStore(path).load_yaml()


def test_load_error_identifies_image_and_field(tmp_path):
    path = tmp_path / "annotations.yaml"
    path.write_text(
        yaml.safe_dump(
            dict(version=2, images={"bad.png": {"shapes": [dict(type="callout", x=0, y=0)]}})
        )
    )
    with pytest.raises(ValueError, match="bad.png.*x2"):
        AnnotationStore(path).load_yaml()


def test_duplicate_number_rejected(tmp_path):
    store = AnnotationStore(tmp_path / "annotations.yaml")
    with pytest.raises(ValueError, match="duplicate callout number"):
        store.add_image("duplicate.png", [callout(), callout()])
    assert store.data == {"version": 1, "images": {}}
