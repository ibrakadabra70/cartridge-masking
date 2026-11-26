"""
Command-line entry point.

Examples (from the project root, with the virtual environment active):

    # every 3DVM image in a folder
    python run.py --input data/samples --output outputs

    # one file
    python run.py --input "data/samples/Fadul 1-1 3DVM.PNG" --output outputs

    # override the pixel size, or use the x3p files for the FPI rim
    python run.py --input data/raw/fadul_2011 --output outputs/full --use-x3p
    python run.py --input my_scan.png --um-per-px 3.12

Outputs (per image, in <output>/<image name>/):
    side_by_side.png   original | overlay, with legend      (for people)
    overlay.png        overlay only, full resolution
    mask_colour.png    coloured labels on black
    labels.png         label image, one value per region    (for programs)
    labels.npy         same as labels.png, as a NumPy array
and for the whole run:
    summary.csv        one row per image (measurements, flags, warnings)
    contact_sheet.png  thumbnails of every overlay
"""

import argparse
import csv
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from src.config import LABELS, Config
from src.io_utils import list_inputs, short_name
from src.pipeline import process
from src.visualize import colour_mask, contact_sheet, overlay, save_rgb, side_by_side


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Automatic colour masking of cartridge case head scans.")
    p.add_argument("--input", "-i", required=True,
                   help="an image file, or a folder of images")
    p.add_argument("--output", "-o", default="outputs",
                   help="folder for results (default: outputs)")
    p.add_argument("--um-per-px", type=float, default=None,
                   help="pixel size in micrometres; overrides the scale bar")
    p.add_argument("--use-x3p", action="store_true",
                   help="use each scan's matching .x3p file (same folder) to "
                        "place the firing pin impression rim")
    p.add_argument("--no-figures", action="store_true",
                   help="skip the side-by-side figures (faster)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    cfg = Config()
    inputs = list_inputs(Path(args.input))
    out_root = Path(args.output)
    out_root.mkdir(parents=True, exist_ok=True)
    print(f"{len(inputs)} image(s) -> {out_root}")

    rows, thumbs, n_fail = [], [], 0
    for k, path in enumerate(inputs, 1):
        name = short_name(path)
        t0 = time.time()
        try:
            result = process(path, cfg, args.um_per_px, args.use_x3p)
            out = out_root / name
            out.mkdir(exist_ok=True)
            ov = overlay(result, cfg)
            save_rgb(out / "overlay.png", ov)
            save_rgb(out / "mask_colour.png", colour_mask(result.labels, cfg))
            save_rgb(out / "labels.png", np.dstack([result.labels] * 3))
            np.save(out / "labels.npy", result.labels)
            if not args.no_figures:
                side_by_side(result, cfg, out / "side_by_side.png")
            row = result.summary()
            thumbs.append((name, ov))
            note = ("drag %.0f deg" % row["drag_angle_deg"]
                    if row["drag_detected"] else "no drag")
            note += ", ap.shear" if row["aperture_shear_detected"] else ""
            print(f"[{k}/{len(inputs)}] {name}: ok ({note}) "
                  f"{time.time() - t0:.1f}s"
                  + (f"  WARNING: {row['warnings']}" if row["warnings"] else ""))
        except Exception as exc:  # one bad image must not stop a batch
            n_fail += 1
            row = {"file": path.name, "status": f"failed: {exc}"}
            thumbs.append((name, None))
            print(f"[{k}/{len(inputs)}] {name}: FAILED ({exc})")
            traceback.print_exc(limit=2)
        rows.append(row)

    # Summary CSV (union of all columns, failures have most cells empty).
    columns = list(dict.fromkeys(c for r in rows for c in r))
    with open(out_root / "summary.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=columns)
        w.writeheader()
        w.writerows(rows)
    if len(thumbs) > 1:
        contact_sheet(thumbs, cfg, out_root / "contact_sheet.png")

    print(f"done: {len(inputs) - n_fail} ok, {n_fail} failed. "
          f"Labels: {LABELS}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
