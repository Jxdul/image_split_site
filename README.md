# Image Split Web UI

Small Flask front-end that embeds the 3:5 splitter logic directly (no external script needed). Upload an image in the browser, split it into panels, preview them, and download individual panels or a ZIP.

## Quick start
1. Optional: create/activate a virtualenv.
2. Install deps (Flask + Pillow):
   ```bash
   python3 -m pip install flask pillow
   ```
3. From `~/image_split_site`, run the server:
   ```bash
   python3 app.py
   ```
4. Open `http://localhost:80` (or your chosen port) in your browser. Upload an image (PNG/JPG/WEBP/BMP/TIFF). Outputs are written into `uploads/<job>/panels/` inside this folder and served back for preview/download.

## How it works
- `/upload` accepts the file, saves it to `uploads/<job>/`, and runs the embedded splitter to produce 3:5 PNG panels into `uploads/<job>/panels/`.
- The produced PNGs are served back through `/file/<job>/<filename>` for inline display.
- `/zip/<job>` streams a ZIP of all panels for that job.
