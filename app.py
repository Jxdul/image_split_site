from __future__ import annotations
import io
import math
import shutil
import uuid
from pathlib import Path
from typing import Dict, List

from flask import Flask, abort, jsonify, render_template, request, send_file, send_from_directory, url_for
from werkzeug.utils import secure_filename
from PIL import Image

# Raise Pillow's decompression bomb limit (use with care).
Image.MAX_IMAGE_PIXELS = 1_000_000_000  # ~1 billion pixels allowed

# Cap each output image to a maximum of 24 megapixels (24,000,000 pixels)
MAX_OUTPUT_PIXELS = 24_000_000

# Choose a high-quality resampler with backwards compatibility
try:
    RESAMPLER = Image.Resampling.LANCZOS
except AttributeError:  # pragma: no cover - older Pillow
    RESAMPLER = Image.LANCZOS

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_ROOT = BASE_DIR / "uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
MAX_UPLOAD_DIRS = 5

app = Flask(__name__)


def _is_allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXTS


def compute_positions_cover(total: int, window: int) -> List[int]:
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
    """Save as PNG, converting mode if necessary and enforcing 24MP cap."""
    original_size = image.size
    image = downscale_to_max_pixels(image, MAX_OUTPUT_PIXELS)
    if image.mode not in ("RGB", "RGBA", "L", "LA"):
        image = image.convert("RGBA")
    image.save(str(path), "PNG")
    if image.size != original_size:
        print(
            f"Resized panel from {original_size[0]}x{original_size[1]} to {image.size[0]}x{image.size[1]} to meet 24MP cap."
        )


def split_cover_3x5_local(input_path: Path, out_dir: Path, prefix: str) -> List[Path]:
    """Split a tall image into 3:5 panels using full width and cover-style spacing."""
    im = Image.open(input_path)
    im.load()

    W, H = im.size
    if W <= 0 or H <= 0:
        raise ValueError("Invalid image dimensions.")

    panel_w = W
    panel_h = round(panel_w * (5 / 3))

    if panel_h <= 0:
        raise ValueError("Computed panel height is non-positive. Unexpected.")
    if H < panel_h:
        raise ValueError(
            f"Image not tall enough for a single 3:5 panel using full width.\n"
            f"Image size: {W}x{H}, required panel height: {panel_h}."
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    positions = compute_positions_cover(H, panel_h)
    digits = max(3, len(str(len(positions))))

    saved_paths: List[Path] = []
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
        dest = out_dir / name
        save_png(out, dest)
        saved_paths.append(dest)

    return saved_paths


def prune_uploads(max_keep: int = MAX_UPLOAD_DIRS) -> None:
    """Keep only the newest max_keep upload directories."""
    try:
        dirs = [d for d in UPLOAD_ROOT.iterdir() if d.is_dir()]
    except FileNotFoundError:
        return

    dirs_sorted = sorted(dirs, key=lambda p: p.stat().st_mtime, reverse=True)
    for old_dir in dirs_sorted[max_keep:]:
        try:
            shutil.rmtree(old_dir)
        except OSError:
            pass


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/upload")
def upload():
    if "image" not in request.files:
        return jsonify({"error": "No file part named 'image'"}), 400

    file = request.files["image"]
    if not file or file.filename == "":
        return jsonify({"error": "Please choose an image to upload."}), 400

    filename = secure_filename(file.filename)
    if not _is_allowed(filename):
        return jsonify({"error": "Unsupported file type."}), 400

    job_id = uuid.uuid4().hex[:12]
    job_dir = UPLOAD_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    upload_path = job_dir / filename
    file.save(upload_path)

    out_dir = job_dir / "panels"
    prefix = f"panel_{job_id[:6]}"
    try:
        panels = split_cover_3x5_local(upload_path, out_dir=out_dir, prefix=prefix)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    prune_uploads()

    image_urls = [url_for("serve_file", job_id=job_id, filename=p.name) for p in sorted(panels)]
    zip_url = url_for("download_zip", job_id=job_id)

    return jsonify({
        "jobId": job_id,
        "images": image_urls,
        "zipUrl": zip_url,
        "count": len(image_urls),
        "targetDir": str(out_dir),
    })


@app.get("/file/<job_id>/<path:filename>")
def serve_file(job_id: str, filename: str):
    out_dir = UPLOAD_ROOT / job_id / "panels"
    file_path = out_dir / filename
    if not file_path.exists() or not file_path.is_file():
        abort(404)
    return send_from_directory(out_dir, filename)


@app.get("/zip/<job_id>")
def download_zip(job_id: str):
    out_dir = UPLOAD_ROOT / job_id / "panels"
    if not out_dir.exists():
        abort(404)

    import zipfile

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for file_path in sorted(out_dir.glob("*.png")):
            if file_path.is_file():
                zf.write(file_path, arcname=file_path.name)
    buffer.seek(0)
    download_name = f"{out_dir.name}.zip"
    return send_file(buffer, mimetype="application/zip", as_attachment=True, download_name=download_name)


def build_job_payload(job_id: str) -> Dict[str, object]:
    """Assemble response payload for a given job id."""
    out_dir = UPLOAD_ROOT / job_id / "panels"
    if not out_dir.exists():
        abort(404)
    panels = sorted([p for p in out_dir.glob("*.png") if p.is_file()])
    if not panels:
        abort(404)
    image_urls = [url_for("serve_file", job_id=job_id, filename=p.name) for p in panels]
    zip_url = url_for("download_zip", job_id=job_id)
    return {
        "jobId": job_id,
        "images": image_urls,
        "zipUrl": zip_url,
        "count": len(image_urls),
        "targetDir": str(out_dir),
    }


@app.get("/job/<job_id>")
def job_info(job_id: str):
    return jsonify(build_job_payload(job_id))


@app.get("/recent")
def recent_job():
    """Return the most recently modified job (if any)."""
    dirs = [d for d in UPLOAD_ROOT.iterdir() if d.is_dir()]
    if not dirs:
        return jsonify({"message": "No jobs yet"}), 404
    newest = max(dirs, key=lambda p: p.stat().st_mtime)
    return jsonify(build_job_payload(newest.name))


if __name__ == "__main__":
    # Use host=0.0.0.0 to be reachable on LAN if needed; change as appropriate.
    app.run(debug=True, host="0.0.0.0", port=6767)
