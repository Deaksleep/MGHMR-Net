#!/usr/bin/env python3
"""Safely extract a downloaded TIFF archive into the training data directory."""

import argparse
import shutil
import zipfile
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/images.zip"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/images"))
    parser.add_argument("--force", action="store_true", help="Overwrite existing TIFF files.")
    return parser.parse_args()


def prepare_data(archive, output_dir, force=False):
    """Extract TIFF members while discarding the archive's top-level folder."""
    if not archive.is_file():
        raise FileNotFoundError(f"Archive not found: {archive}")
    output_dir.mkdir(parents=True, exist_ok=True)

    extracted = 0
    skipped = 0
    with zipfile.ZipFile(archive) as zip_file:
        for member in zip_file.infolist():
            member_path = Path(member.filename)
            if member.is_dir() or member_path.suffix.lower() not in {".tif", ".tiff"}:
                continue
            destination = output_dir / member_path.name
            if destination.exists() and not force:
                skipped += 1
                continue
            # Copy a single known-safe basename; never trust archive directory paths.
            with zip_file.open(member) as source, destination.open("wb") as target:
                shutil.copyfileobj(source, target)
            extracted += 1

    total = len(list(output_dir.glob("*.tif"))) + len(list(output_dir.glob("*.tiff")))
    if total == 0:
        raise RuntimeError(f"No TIFF images were found in {archive}")
    print(f"Prepared {total} images in {output_dir} ({extracted} extracted, {skipped} skipped).")


if __name__ == "__main__":
    cli_args = parse_args()
    prepare_data(cli_args.archive, cli_args.output_dir, cli_args.force)
