import shutil
from pathlib import Path


def copy_plotted_spectra(src_path: Path, target_path: Path) -> None:
    for f in src_path.rglob("*spectra.*"):
        dest = target_path / f.name
        print(f"[INFO] {f} -> {dest}")
        shutil.copy(f, dest)


def copy_msa_files(src_dir: Path, target_dir: Path) -> None:
    """Copy only .msa files from src_dir into target_dir."""
    target_dir.mkdir(exist_ok=True, parents=True)

    for f in src_dir.rglob("*.msa"):
        dest_dir = target_dir / f.parent.name
        dest_dir.mkdir(exist_ok=True, parents=True)
        dest = dest_dir / f.name
        print(f"{f} -> {dest}")
        shutil.copy(f, dest)


def copy_tex_files(src_dir: Path, target_dir: Path) -> None:
    """Copy only .msa files from src_dir into target_dir."""
    target_dir.mkdir(exist_ok=True, parents=True)

    for f in src_dir.rglob("*.tex"):
        dest_dir = target_dir / f.parent.name
        dest_dir.mkdir(exist_ok=True, parents=True)
        dest = dest_dir / f.name
        print(f"{f} -> {dest}")
        shutil.copy(f, dest)


def copy_results(src_path: Path, target_path: Path) -> None:
    """Copy spectra files from src_path to target_path."""
    if not src_path.exists():
        raise ValueError(f"Source path does not exist: {src_path}")
    target_path.mkdir(exist_ok=True, parents=True)
    copy_plotted_spectra(src_path, target_path)
    copy_msa_files(src_path, target_path)


if __name__ == "__main__":
    copy_results(
        Path("/home/m/Dropbox/ZUMI/zakazky/ZADANI-SEM/sem-25-08/"),
        Path("/home/m/Dropbox/ZUMI/zakazky/2511_Szabová/pytex/sem"),
    )
