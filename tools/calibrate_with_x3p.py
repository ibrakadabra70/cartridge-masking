"""
Calibrate Config.fpi_rim_fraction from the 3D topography (x3p files).

The shaded PNG barely shows where the firing pin crater ends. The x3p height
maps do: the rim is the crest of the raised flow-back ring. This script
measures the crest on every x3p file in a folder (see src/topography.py for
the method) and prints the median rim fraction, which is the default used
for PNG-only processing.

Usage (from the project root):
    python tools/calibrate_with_x3p.py data/raw/fadul_2011
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import Config                      # noqa: E402
from src.topography import rim_profile_from_x3p    # noqa: E402


def main(folder: str):
    cfg = Config()
    fractions = []
    print(f"{'file':14s} {'core':>6s} {'crest':>6s} {'apert':>6s} "
          f"{'ap/core':>7s} {'f':>5s}   (radii in x3p pixels)")
    for path in sorted(Path(folder).glob("*.x3p")):
        try:
            i = rim_profile_from_x3p(path, cfg)
        except Exception as exc:  # report and continue
            print(f"{path.stem:14s} FAILED: {exc}")
            continue
        fractions.append(i["rim_fraction"])
        print(f"{path.stem:14s} {i['core_r']:6.0f} {i['crest_r']:6.0f} "
              f"{i['aperture_r']:6.0f} {i['aperture_r'] / i['core_r']:7.2f} "
              f"{i['rim_fraction']:5.2f}")
    if not fractions:
        print("No x3p files processed.")
        return
    q = np.percentile(fractions, [10, 50, 90])
    print(f"\nn = {len(fractions)}   rim fraction f: p10 = {q[0]:.2f}  "
          f"median = {q[1]:.2f}  p90 = {q[2]:.2f}")
    print("-> Config.fpi_rim_fraction should be set to the median.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/raw/fadul_2011")
