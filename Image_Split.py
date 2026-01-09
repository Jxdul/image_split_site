#!/usr/bin/env python3
"""
-Jadyl Posadas
Split a long image into multiple equal 3:5 (width:height) panels using cover-style placement,
and always save outputs into your ~/Downloads directory (by default into a subfolder).

- Supports input formats handled by Pillow (PNG, JPEG/JPG, WEBP, BMP, TIFF, etc.).
- **Always outputs PNG** files.
- Uses the full source image width for each panel.
- Panel height = round(width * 5 / 3).
- Panels are positioned so the first starts at y=0 and the last ends exactly at the bottom.
- If rounding causes a tiny short crop at the bottom, it pads to keep exact size.
- Outputs go to: ~/Downloads/<input_stem>_3x5  (unless --flat is used)

Usage:
  python Image_Split.py input_image [input_image ...]
  python Image_Split.py input_image [input_image ...] --prefix mycut --flat

Requires:
  pip install pillow
"""

import argparse
import math
import os
from pathlib import Path
from PIL import Image
# Raise Pillow's decompression bomb limit (use with care).
Image.MAX_IMAGE_PIXELS = 1_000_000_000  # ~1 billion pixels allowed

# Cap each output image to a maximum of 24 megapixels (24,000,000 pixels)
MAX_OUTPUT_PIXELS = 24_000_000

# Choose a high-quality resampler with backwards compatibility
try:
    RESAMPLER = Image.Resampling.LANCZOS
except AttributeError:
    RESAMPLER = Image.LANCZOS

def compute_positions_cover(total: int, window: int) -> list[int]:
    """Evenly space crop tops so first starts at 0 and last ends at total."""
    if total <= window:
        return [0]
    n = math.ceil(total / window)
    max_start = total - window
    positions = [round(i * (max_start / (n - 1))) for i in range(n)]
    # guard against rounding collisions
    for i in range(1, len(positions)):
        if positions[i] <= positions[i - 1]:
            positions[i] = positions[i - 1] + 1
    positions[-1] = max_start
    return positions


def get_downloads_dir() -> Path:
    """Best-effort: use ~/Downloads, create it if missing."""
    d = Path.home() / "Downloads"
    d.mkdir(parents=True, exist_ok=True)
    return d

def find_most_recent_image(directory: Path) -> Path:
    """Return the most recently modified image file in the given directory.

    Considers common image types supported by Pillow. Non-recursive.
    Raises FileNotFoundError if none are found.
    """
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
    latest_path = None
    latest_mtime = -1.0

    if not directory.exists() or not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    for p in directory.iterdir():
        if not p.is_file():
            continue
        # Skip temp/incomplete downloads
        if p.name.startswith(".") or p.suffix.lower() not in exts or p.suffix.endswith("crdownload"):
            if p.suffix.lower().endswith("crdownload"):
                continue
            if p.suffix.lower() not in exts:
                continue
        try:
            mtime = p.stat().st_mtime
        except OSError:
            continue
        if mtime > latest_mtime:
            latest_mtime = mtime
            latest_path = p

    if latest_path is None:
        raise FileNotFoundError("No image files found in the directory.")

    return latest_path


def downscale_to_max_pixels(im: Image.Image, max_pixels: int) -> Image.Image:
    """If the image exceeds max_pixels in area, downscale it proportionally."""
    w, h = im.size
    area = w * h
    if area <= max_pixels:
        return im
    scale = math.sqrt(max_pixels / area)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    return im.resize((new_w, new_h), RESAMPLER)

def save_png(image: Image.Image, path: Path) -> None:
    """
    Save as PNG, converting mode if necessary. Preserves transparency when present.
    """
    # Enforce 24MP cap on output size
    original_size = image.size
    image = downscale_to_max_pixels(image, MAX_OUTPUT_PIXELS)
    if image.size != original_size:
        try:
            print(f"Resized panel from {original_size[0]}x{original_size[1]} to {image.size[0]}x{image.size[1]} to meet 24MP cap.")
        except Exception:
            pass
    # Convert uncommon modes to RGBA for consistent PNG output
    if image.mode not in ("RGB", "RGBA", "L", "LA"):
        image = image.convert("RGBA")
    image.save(str(path), "PNG")

def split_cover_3x5(input_path: str, prefix: str = "panel", flat: bool = False) -> Path:
    im = Image.open(input_path)
    im.load()

    W, H = im.size
    if W <= 0 or H <= 0:
        raise ValueError("Invalid image dimensions.")

    # 3:5 aspect using full width
    panel_w = W
    panel_h = round(panel_w * (5 / 3))

    if panel_h <= 0:
        raise ValueError("Computed panel height is non-positive. Unexpected.")
    if H < panel_h:
        raise ValueError(
            f"Image not tall enough for a single 3:5 panel using full width.\n"
            f"Image size: {W}x{H}, required panel height: {panel_h}."
        )

    downloads = get_downloads_dir()
    stem = Path(input_path).stem
    out_dir = downloads if flat else downloads / f"{stem}_3x5"
    out_dir.mkdir(parents=True, exist_ok=True)

    positions = compute_positions_cover(H, panel_h)
    digits = max(3, len(str(len(positions))))

    saved = 0
    for idx, top in enumerate(positions, start=1):
        top = int(top)
        bottom = min(H, top + panel_h)
        crop = im.crop((0, top, panel_w, bottom))

        # Pad if the last crop is short due to rounding
        if crop.size != (panel_w, panel_h):
            out = Image.new("RGBA", (panel_w, panel_h), (0, 0, 0, 0))
            out.paste(crop, (0, 0))
        else:
            out = crop

        name = f"{prefix}_{idx:0{digits}d}.png"
        save_png(out, out_dir / name)
        saved += 1

    print(f"Done. {input_path} ({W}x{H}) → {saved} PNG panels of {panel_w}x{panel_h} (3:5) in '{out_dir}'.")
    return out_dir

def main():
    ap = argparse.ArgumentParser(description="Split tall images into equal 3:5 panels (cover-only) into ~/Downloads. If no inputs are given, the most recent image in ~/Downloads is used.")
    ap.add_argument("inputs", nargs="*", help="Path(s) to input image(s) (tall/long). Supports PNG/JPEG/WEBP/BMP/TIFF; outputs PNG. If omitted, the most recent image in ~/Downloads is used.")
    ap.add_argument("--prefix", default="panel", help="Output filename prefix (default: panel).")
    ap.add_argument("--flat", action="store_true",
                    help="Write files directly into ~/Downloads (no subfolder).")
    args = ap.parse_args()

    if not args.inputs:
        try:
            latest = find_most_recent_image(get_downloads_dir())
            args.inputs = [str(latest)]
            print(f"No inputs given; using most recent image in ~/Downloads: {latest.name}")
        except FileNotFoundError:
            print("No inputs provided and no images found in ~/Downloads.")
            return

    any_errors = False
    for in_path in args.inputs:
        try:
            split_cover_3x5(
                input_path=in_path,
                prefix=args.prefix,
                flat=args.flat,
            )
        except Exception as e:
            any_errors = True
            print(f"Error processing '{in_path}': {e}")
    if any_errors:
        print("Finished with some errors (see above).")
    else:
        print("All images processed successfully.")

if __name__ == "__main__":
    main()