#!/usr/bin/env python3
import argparse
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Tuple, Optional, List

from PIL import Image, ImageOps

# EXIF tag ids
EXIF_ORIENTATION = 274
EXIF_DATE_TIME = 306               # "YYYY:MM:DD HH:MM:SS"
EXIF_DATE_TIME_ORIGINAL = 36867    # "YYYY:MM:DD HH:MM:SS"
EXIF_DATE_TIME_DIGITIZED = 36868   # "YYYY:MM:DD HH:MM:SS"

# e.g., "60img105.jpg" or "60IMG105.tif"
NAME_RE = re.compile(r"^(\d{2})img(\d{3})", re.IGNORECASE)


def infer_century_from_folder(folder: Path) -> Optional[int]:
    name = folder.name.lower()
    if "190" in name or "1900" in name or "1900s" in name or "19xx" in name:
        return 1900
    if "200" in name or "2000" in name or "2000s" in name or "20xx" in name:
        return 2000
    return None


def parse_filename(stem: str) -> Tuple[int, int]:
    """
    Returns (yy, idx) from a stem like '60img105'.
    yy: 0..99 two-digit year
    idx: 0..999 within-year ordering index
    """
    m = NAME_RE.match(stem)
    if not m:
        raise ValueError(f"Filename stem does not match 'YYimgNNN' pattern: '{stem}'")
    yy = int(m.group(1))
    idx = int(m.group(2))
    return yy, idx


def exif_dt_string(dt: datetime) -> str:
    return dt.strftime("%Y:%m:%d %H:%M:%S")


def write_exif_and_fix_orientation(img_path: Path, dt: datetime, dry_run: bool = False) -> None:
    """
    - Apply EXIF orientation to pixels
    - Remove orientation flag
    - Write EXIF dates to dt
    - Save in-place (preserving format)
    """
    with Image.open(img_path) as im:
        # Apply orientation
        im_corrected = ImageOps.exif_transpose(im)

        # Update EXIF
        exif = im.getexif()
        if EXIF_ORIENTATION in exif:
            del exif[EXIF_ORIENTATION]

        dt_str = exif_dt_string(dt)
        exif[EXIF_DATE_TIME] = dt_str
        exif[EXIF_DATE_TIME_ORIGINAL] = dt_str
        exif[EXIF_DATE_TIME_DIGITIZED] = dt_str

        if dry_run:
            print(f"[DRY-RUN] Would set EXIF times for '{img_path.name}' -> {dt_str}")
            return

        save_kwargs = {}
        try:
            exif_bytes = exif.tobytes()
            if exif_bytes:
                save_kwargs["exif"] = exif_bytes
        except Exception:
            pass

        # Save back to the same path (JPEG will recompress)
        im_corrected.save(img_path, **save_kwargs)


def main():
    parser = argparse.ArgumentParser(
        description="Set EXIF year+ordering from 'YYimgNNN' filenames, fix EXIF rotation, and rename by time."
    )
    parser.add_argument("folder", type=Path, help="Folder with images named like '60img105.jpg'.")
    parser.add_argument("--century", choices=["1900", "2000", "auto"], default="auto",
                        help="Century for 2-digit years (default: auto from folder name).")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without writing.")
    parser.add_argument("--start-index", type=int, default=1, help="Starting 'N' for renaming (default: 1).")
    parser.add_argument("--strip-existing-prefix", action="store_true",
                        help="Remove any leading '<digits>-' before adding the new index.")
    args = parser.parse_args()

    folder: Path = args.folder
    if not folder.is_dir():
        raise SystemExit(f"Not a directory: {folder}")

    # Century
    if args.century == "auto":
        century = infer_century_from_folder(folder)
        if century is None:
            raise SystemExit("Could not infer century from folder name. Use --century 1900 or --century 2000.")
    else:
        century = int(args.century)

    # Collect files
    files: List[Path] = sorted(p for p in folder.iterdir() if p.is_file())
    if not files:
        raise SystemExit("No files found.")

    # Build records with computed datetime (Jan 1 + index seconds)
    records = []
    for p in files:
        try:
            yy, idx = parse_filename(p.stem)
            year_full = century + yy
            dt = datetime(year_full, 1, 1, 0, 0, 0) + timedelta(seconds=idx)
            write_exif_and_fix_orientation(p, dt, dry_run=args.dry_run)
            records.append((p, dt))
        except Exception as e:
            print(f"[WARN] Skipping '{p.name}': {e}")

    if not records:
        raise SystemExit("No valid images processed.")

    # Sort by datetime, then by name as tiebreaker
    records.sort(key=lambda t: (t[1], t[0].name.lower()))

    # Rename to N-<original>
    idx = args.start_index
    for path, dt in records:
        base = path.name
        if args.strip_existing_prefix:
            base = re.sub(r"^\d+-", "", base)

        new_name = f"{idx}-{base}"
        new_path = path.with_name(new_name)

        # Ensure uniqueness (unlikely)
        while new_path.exists():
            idx += 1
            new_name = f"{idx}-{base}"
            new_path = path.with_name(new_name)

        if args.dry_run:
            print(f"[DRY-RUN] Would rename '{path.name}' -> '{new_name}' at {exif_dt_string(dt)}")
        else:
            path.rename(new_path)
            print(f"Renamed: '{path.name}' -> '{new_name}'")

        idx += 1


if __name__ == "__main__":
    main()
