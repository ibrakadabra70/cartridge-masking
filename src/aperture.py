"""
Find the firing-pin APERTURE edge.

What we are looking for
-----------------------
The breech face has a round hole (the firing-pin aperture). When the gun
fires, primer metal flows back into this hole. The rim of the hole leaves a
steep step that the scanner cannot measure, so the PNG shows a thin, WHITE,
BROKEN arc between the textured breech-face ring and the inner flow-back.

The arc is broken (sometimes more than half of it is missing) and white
speckle exists everywhere, so tracing it pixel by pixel is fragile.
Instead we use two strong priors:

* the aperture is a CIRCLE, and
* it is roughly CONCENTRIC with the firing-pin crater (the pin travels through
  the hole), with a radius of about 1.3-2x the crater's core radius.

Method: RANSAC circle search with an angular-coverage score
-----------------------------------------------------------
1. Candidate pixels = white (no-data) pixels inside the primer, minus the
   central firing-pin blob and tiny speckles.
2. Repeat many times: pick 3 random candidates, compute the circle through
   them, reject it unless its centre is near the blob centre and its radius
   is in the expected range.
3. Score = in how many of 360 angular bins (around the circle's centre)
   a candidate pixel lies ON the circle (within a thin band). A real arc
   covers many angles at one consistent radius; random speckle or a local
   clump does not. This is why broken arcs still win.
4. Refine the best circle by least squares on its inliers.

Rotation invariance: nothing here depends on direction, only on radii.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .firing_pin import Blob
from .geometry import (Circle, circle_from_3_points, disc_mask, fit_circle_lsq,
                       morph, remove_small)
from .primer import Primer


@dataclass
class Aperture:
    circle: Circle
    coverage: float           # fraction of the circle supported by white arc
    arc_pixels: np.ndarray    # bool mask of the white pixels on the circle


def _coverage(xs, ys, circle: Circle, band: float, bins: int):
    """Inlier mask and fraction of angular bins hit by inliers."""
    d = np.hypot(xs - circle.cx, ys - circle.cy)
    inl = np.abs(d - circle.r) < band
    if not inl.any():
        return inl, 0.0
    ang = np.arctan2(ys[inl] - circle.cy, xs[inl] - circle.cx)
    hit = np.unique(((ang + np.pi) / (2 * np.pi) * bins).astype(int) % bins)
    return inl, hit.size / bins


def detect_aperture(primer: Primer, blob: Blob, cfg: Config,
                    rng: np.random.Generator | None = None) -> Aperture:
    rng = rng or np.random.default_rng(0)  # fixed seed -> reproducible
    R = primer.circle.r
    core = blob.core

    # Candidates: white pixels in the ANNULUS where the aperture can be.
    # (We deliberately do not restrict to the primer region: on off-centre
    # scans the arc merges with the white background outside the primer.)
    r_lo, r_hi = (f * core.r for f in cfg.aperture_r_over_core)
    pad = cfg.aperture_max_offset * R
    outer = disc_mask(primer.measured.shape, Circle(core.cx, core.cy, r_hi + pad))
    inner = disc_mask(primer.measured.shape, Circle(core.cx, core.cy,
                                                    max(r_lo - pad, 0)))
    white = ~primer.measured & outer & ~inner
    # Exclude the blob (slightly grown, to drop its ragged edge).
    white &= ~morph(blob.mask, cv2.MORPH_DILATE, 0.03 * R)
    cand = remove_small(white, cfg.aperture_min_speckle_area * R * R)

    ys, xs = np.nonzero(cand)
    if xs.size < 30:
        raise RuntimeError("Aperture: not enough white arc pixels")
    if xs.size > 8000:  # subsample for speed; the score is a fraction
        idx = rng.choice(xs.size, 8000, replace=False)
        xs, ys = xs[idx], ys[idx]
    xs, ys = xs.astype(float), ys.astype(float)

    band = cfg.aperture_band * R
    bins = cfg.aperture_angle_bins
    max_off = cfg.aperture_max_offset * R

    def plausible(c: Circle) -> bool:
        return (r_lo <= c.r <= r_hi and
                np.hypot(c.cx - core.cx, c.cy - core.cy) <= max_off)

    best, best_score = None, -1.0
    for _ in range(cfg.aperture_ransac_iters):
        i = rng.choice(xs.size, 3, replace=False)
        circ = circle_from_3_points(list(zip(xs[i], ys[i])))
        if circ is None or not plausible(circ):
            continue
        _, score = _coverage(xs, ys, circ, band, bins)
        if score > best_score:
            best, best_score = circ, score

    if best is None:
        raise RuntimeError("Aperture: no plausible circle found")

    # Least-squares refinement on the inliers (a few rounds).
    for _ in range(3):
        inl, _ = _coverage(xs, ys, best, band * 1.5, bins)
        if inl.sum() < 10:
            break
        refined = fit_circle_lsq(xs[inl], ys[inl])
        if not plausible(refined):
            break
        best = refined
    _, coverage = _coverage(xs, ys, best, band, bins)

    # White pixels on the final circle (for display / debugging).
    yy, xx = np.nonzero(cand)
    on = np.abs(np.hypot(xx - best.cx, yy - best.cy) - best.r) < band * 1.5
    arc = np.zeros_like(cand)
    arc[yy[on], xx[on]] = True
    return Aperture(best, float(coverage), arc)
