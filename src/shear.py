"""
Aperture shear and firing pin shear.

Both are STRETCH goals. They are detected with simpler, weaker evidence than
the breech face and firing pin impression, so read the confidence notes in
README.md before relying on them.

Aperture shear (green)
----------------------
During unlocking, the barrel/slide moves relative to the cartridge while the
primer metal is still pushed into the firing-pin aperture. The aperture's
edge scrapes that flow-back and leaves PARALLEL STRIATIONS near the aperture
edge, roughly perpendicular to the edge, i.e. running RADIALLY.

In the 3D rendering these striations show up as short WHITE DASHES (the
scanner loses the steep striation walls) separated by measured metal, which
form a "ladder" between the flow-back and the aperture edge.

Detection, done in polar coordinates around the aperture centre so that it
is rotation invariant:

1. Unwrap the shear band (from the FPI rim out to just past the aperture
   edge) into a (radius x angle) image. Radial striations become VERTICAL
   streaks; the aperture arc becomes a roughly HORIZONTAL white line.
2. For every angle, count white <-> grey transitions when stepping along
   the angle direction, averaged over all radii. Vertical streaks produce
   many transitions; a horizontal arc produces almost none, so the arc
   itself does not trigger the detector.
3. Smooth this "dash density" profile D(theta) over angle. The shear is the
   angular SECTOR around the strongest peak, if that peak clearly stands
   out from the scan's own typical level (median). If a firing pin drag was
   found, the peak is searched only on the half of the circle facing away
   from the drag (where the trailing aperture edge scrapes the primer).

(An earlier version used gradient-energy anisotropy; it was dominated by the
lighting of the rendering and by the arc, so it was replaced.)

Result: a crescent-shaped sector of the band. The method localises WHERE
along the aperture the striations are, not each individual striation.

Firing pin shear (purple) -- follows the assignment's reference figure
---------------------------------------------------------------------
In the reference annotation (assignment PDF, page 2) the elongated firing
pin impression is split in two: the ROUND, TRAILING part, where the pin
first struck and then sheared away from, is purple ("firing pin shear"), and
the part that extends towards the drag arrow is light blue ("firing pin
impression"). We reproduce that convention geometrically:

    purple = the round core of the crater, on the side of the line through
             its centre (perpendicular to the drag) that faces AWAY from
             the drag.

It exists only if a drag was detected; otherwise the whole crater is the
firing pin impression. This is a geometric split that copies the reference
convention, not a texture measurement.
"""

from dataclasses import dataclass

import numpy as np

from .aperture import Aperture
from .config import Config
from .firing_pin import FiringPin
from .geometry import angle_diff, disc_mask, polar_coords, unwrap_polar
from .primer import Primer


@dataclass
class ApertureShear:
    mask: np.ndarray
    detected: bool
    profile: np.ndarray          # dash density D(theta), smoothed
    peak_deg: float | None
    sector_deg: tuple | None     # (start, end), counter-clockwise
    contrast: float              # peak / median of the profile


def _dash_density(polar: np.ndarray, white_level: int) -> np.ndarray:
    """Per angle: mean number of white<->grey transitions between this
    column and the next (0..1), over all radii of the polar band."""
    white = polar >= white_level
    change = white != np.roll(white, -1, axis=1)   # wraps around 360 deg
    return change.mean(axis=0).astype(np.float64)


def _smooth_circular(x: np.ndarray, width: int) -> np.ndarray:
    """Moving average that wraps around (the profile is periodic)."""
    width = max(1, int(width) | 1)
    h = width // 2
    padded = np.r_[x[-h:], x, x[:h]] if h else x
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def detect_aperture_shear(gray: np.ndarray, primer: Primer,
                          aperture: Aperture, fp: FiringPin,
                          cfg: Config) -> ApertureShear:
    R = primer.circle.r
    ap = aperture.circle
    n = cfg.shear_n_theta
    deg_per_bin = 360.0 / n

    r_in, r_out = fp.rim.r, ap.r + cfg.shear_band_outer * R
    band = unwrap_polar(gray, ap.cx, ap.cy, r_in, r_out, n)
    S = _dash_density(band, cfg.no_data_threshold)
    S = _smooth_circular(S, cfg.shear_smooth_deg / deg_per_bin)

    theta = np.arange(n) * deg_per_bin
    allowed = np.ones(n, bool)
    if fp.drag_detected:
        opposite = (fp.drag_angle_deg + 180.0) % 360
        allowed = np.abs(angle_diff(np.radians(theta),
                                    np.radians(opposite))) <= \
            np.radians(cfg.shear_search_half_deg)

    med = float(np.median(S)) + 1e-6
    peak = int(np.argmax(np.where(allowed, S, -np.inf)))
    contrast = float(S[peak] / med)              # peak relative to typical

    empty = np.zeros(gray.shape, bool)
    if contrast < cfg.shear_min_contrast:
        return ApertureShear(empty, False, S, None, None, contrast)

    # Grow the sector from the peak while S stays above half height.
    level = med + cfg.shear_sector_level * (S[peak] - med)
    max_half = int(cfg.shear_max_sector_deg / 2 / deg_per_bin)
    lo = hi = 0
    while lo < max_half and S[(peak - lo - 1) % n] >= level:
        lo += 1
    while hi < max_half and S[(peak + hi + 1) % n] >= level:
        hi += 1
    min_half = int(cfg.shear_min_sector_deg / 2 / deg_per_bin)
    lo, hi = max(lo, min_half), max(hi, min_half)
    start, end = (peak - lo) * deg_per_bin, (peak + hi) * deg_per_bin

    # Build the image mask: band radii x sector angles.
    r, th = polar_coords(gray.shape, ap.cx, ap.cy)
    th_deg = np.degrees(th) % 360
    in_sector = ((th_deg - start) % 360) <= (end - start)
    mask = (r >= r_in) & (r <= r_out) & in_sector & ~fp.blob
    return ApertureShear(mask, True, S, peak * deg_per_bin,
                         (start % 360, end % 360), contrast)


def detect_firing_pin_shear(shape, aperture: Aperture, fp: FiringPin,
                            cfg: Config) -> np.ndarray:
    """Trailing half of the round crater core (empty if no drag)."""
    if not fp.drag_detected:
        return np.zeros(shape, bool)
    core = fp.core
    r, th = polar_coords(shape, core.cx, core.cy)
    drag = np.radians(fp.drag_angle_deg)
    # |angle from the drag| > 90 deg  <=>  behind the perpendicular line.
    behind = np.abs(angle_diff(th, drag)) > np.radians(90.0)
    return behind & (r <= core.r) & disc_mask(shape, aperture.circle)
