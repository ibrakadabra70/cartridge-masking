"""
Generalisation check: rotation and scale invariance.

The Fadul set has only one calibre (9 mm Luger), so a different calibre
cannot be tested directly. What changes between calibres, guns and scanners
is mostly (a) the size of the primer in pixels (primer diameter, pixel size)
and (b) the orientation of the marks. This script simulates both:

* every scan is ROTATED by several angles   -> drag / shear angles must
  rotate by the same amount, sizes must not change;
* every scan is RESCALED (0.6x ... 1.5x)    -> sizes in mm must stay the
  same, because all parameters are fractions of the primer radius.

It prints the errors and saves a figure of a few transformed overlays.

Usage (from the project root):
    python tools/robustness_check.py data/samples outputs/robustness
"""

import sys
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config                       # noqa: E402
from src.io_utils import list_inputs, load_scan     # noqa: E402
from src.pipeline import process_scan               # noqa: E402
from src.visualize import overlay, save_rgb         # noqa: E402

ANGLES = [0, 45, 90, 137, 200, 290]
SCALES = [0.6, 0.8, 1.0, 1.25, 1.5]


def transform(gray: np.ndarray, angle: float, scale: float) -> np.ndarray:
    """Rotate (counter-clockwise, degrees) and scale an image about its
    centre, padding with white (= no data) so nothing is cut off."""
    h, w = gray.shape
    pad = int(0.25 * max(h, w))
    g = cv2.copyMakeBorder(gray, pad, pad, pad, pad, cv2.BORDER_CONSTANT,
                           value=255)
    H, W = g.shape
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, scale)
    out_size = (int(W * scale), int(H * scale))
    M[:, 2] += (np.array(out_size) - (W, H)) / 2
    return cv2.warpAffine(g, M, out_size, flags=cv2.INTER_LINEAR,
                          borderValue=255)


def ang_err(a, b):
    return abs((a - b + 180) % 360 - 180)


def main(folder: str, out: str):
    cfg = Config()
    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, tiles = [], []
    for path in list_inputs(Path(folder)):
        scan = load_scan(path, cfg)
        ref_res = process_scan(scan, cfg)
        ref, ref_peak = ref_res.summary(), ref_res.aperture_shear.peak_deg
        for angle in ANGLES:
            for scale in SCALES:
                if angle and scale != 1.0:
                    continue  # rotations at scale 1, scales at angle 0
                g = transform(scan.gray, angle, scale)
                s2 = replace(scan, gray=g, um_per_px=scan.um_per_px / scale)
                try:
                    res = process_scan(s2, cfg)
                except Exception as exc:
                    print(f"{scan.name} rot {angle} scale {scale}: FAILED {exc}")
                    rows.append(dict(failed=True))
                    continue
                r = res.summary()
                row = dict(
                    failed=False,
                    primer_mm=abs(r["primer_radius_mm"] - ref["primer_radius_mm"]),
                    rim_mm=abs(r["fpi_rim_radius_mm"] - ref["fpi_rim_radius_mm"]),
                    drag_same=r["drag_detected"] == ref["drag_detected"],
                    drag_deg=(ang_err(r["drag_angle_deg"],
                                      ref["drag_angle_deg"] + angle)
                              if r["drag_detected"] and ref["drag_detected"]
                              else np.nan),
                    shear_deg=(ang_err(res.aperture_shear.peak_deg,
                                       ref_peak + angle)
                               if ref_peak is not None
                               and res.aperture_shear.peak_deg is not None
                               else np.nan),
                )
                rows.append(row)
                print(f"{scan.name:10s} rot {angle:3d} scale {scale:4.2f}: "
                      f"primer err {row['primer_mm']*1000:5.1f} um, "
                      f"rim err {row['rim_mm']*1000:5.1f} um, "
                      f"drag err {row['drag_deg']:5.1f} deg, "
                      f"shear err {row['shear_deg']:5.1f} deg")
                if (angle, scale) in [(0, 1.0), (137, 1.0), (290, 1.0), (0, 0.6)] \
                        and len(tiles) < 8:
                    t = cv2.resize(overlay(res, cfg), (400, 400),
                                   interpolation=cv2.INTER_AREA)
                    cv2.putText(t, f"{scan.name} rot {angle} x{scale}", (8, 26),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 2)
                    tiles.append(t)

    ok = [r for r in rows if not r["failed"]]
    print(f"\n{len(ok)}/{len(rows)} transformed scans processed")
    if ok:
        def stat(k):
            v = np.array([r[k] for r in ok], float)
            v = v[~np.isnan(v)]
            return f"median {np.median(v):.3g}, max {np.max(v):.3g}" if v.size else "n/a"
        print("primer radius error (mm):", stat("primer_mm"))
        print("FPI rim radius error (mm):", stat("rim_mm"))
        print("drag angle error (deg):  ", stat("drag_deg"))
        print("shear peak error (deg):  ", stat("shear_deg"))
        print("drag yes/no unchanged:    "
              f"{sum(r['drag_same'] for r in ok)}/{len(ok)}")
    if tiles:
        while len(tiles) % 4:
            tiles.append(np.full_like(tiles[0], 255))
        grid = np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)])
        save_rgb(out_dir / "robustness_examples.png", grid)
        print(f"figure: {out_dir / 'robustness_examples.png'}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/samples",
         sys.argv[2] if len(sys.argv) > 2 else "outputs/robustness")
