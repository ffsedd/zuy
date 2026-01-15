from pathlib import Path
from PIL import Image
from zuy.common.logger import setup_logger

logger = setup_logger(__name__)


def convert_tiff_to_jpg(root: Path, outdir: Path, overwrite: bool = False) -> None:
    """
    Recursively convert all .tif/.tiff images to .jpg
    """
    for f in root.rglob("*"):
        if f.suffix.lower() in {".tif", ".tiff"}:
            jpg_path = outdir / f"{f.stem}.jpg"
            if jpg_path.exists() and not overwrite:
                logger.info(f"Skipping existing JPG: {jpg_path}")
                continue
            try:
                with Image.open(f) as im:
                    rgb = im.convert("RGB")
                    rgb.save(jpg_path, "JPEG", quality=95)
                logger.info(f"Converted {f} → {jpg_path}")
            except Exception as e:
                logger.warning(f"Failed to convert {f}: {e}")
