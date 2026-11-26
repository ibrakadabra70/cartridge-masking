"""
Using the 3D topography (x3p) to locate the firing-pin-impression rim.

In the shaded PNG the crest of the crater (the real FPI boundary) is hard to
see, but in the height map it is simply the highest ring between the white
blob and the aperture edge (see the sketch in firing_pin.py).

We run the SAME 2D detection on the x3p (missing points play the role of the
white no-data pixels), then:

    1. median height in thin rings around the blob's core centre,
       from the core radius out to the aperture radius;
    2. crest radius = ring with the highest median;
    3. rim fraction f = (r_crest - r_core) / (r_aperture - r_core).

f has no units, so it can be transferred to the PNG (which has a different
pixel size and is not pixel-aligned with the x3p).

Used in two places:
* tools/calibrate_with_x3p.py: median f over a whole data set
  -> the default Config.fpi_rim_fraction;
* run.py --use-x3p: each scan uses its OWN f from its matching x3p file.
"""

from pathlib import Path

import numpy as np

from .aperture import detect_aperture
from .config import Config
from .firing_pin import detect_blob
from .geometry import polar_coords
from .io_utils import read_x3p, x3p_as_gray
from .primer import detect_primer


def rim_profile_from_x3p(path: Path, cfg: Config) -> dict:
    """Detect geometry on the x3p and return the radial height profile."""
    x3p = read_x3p(path)
    gray = x3p_as_gray(x3p)
    primer = detect_primer(gray, cfg)
    blob = detect_blob(gray, primer, cfg)
    ap = detect_aperture(primer, blob, cfg).circle
    core = blob.core

    r, _ = polar_coords(gray.shape, core.cx, core.cy)
    h = x3p.heights_um
    edges = np.arange(core.r, ap.r * 0.97, 2.0)      # 2-px rings
    med = np.full(len(edges) - 1, np.nan)
    for i, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        vals = h[(r >= a) & (r < b)]
        vals = vals[~np.isnan(vals)]
        if vals.size > 50:
            med[i] = np.median(vals)
    if np.all(np.isnan(med)):
        raise RuntimeError("no height data between blob and aperture")
    crest_r = float(edges[np.nanargmax(med)] + 1.0)
    return dict(core_r=core.r, aperture_r=ap.r, crest_r=crest_r,
                rim_fraction=(crest_r - core.r) / (ap.r - core.r),
                radii=(edges[:-1] + 1.0), median_height_um=med)


def rim_fraction_from_x3p(path: Path, cfg: Config) -> float:
    return float(rim_profile_from_x3p(path, cfg)["rim_fraction"])
