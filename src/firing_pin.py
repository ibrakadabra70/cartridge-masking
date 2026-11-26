r"""
Firing pin impression (FPI) and firing pin drag.

Physical picture (confirmed on the x3p height maps)
----------------------------------------------------
Inside the aperture, primer metal bulges up into a raised ring ("flow-back").
The firing pin punched a crater into the middle of it:

    height
      ^          crest                     crest
      |         .-'''-.                   .-'''-.
      |   BF   /       \     crater      /       \   BF
      |_______/         \               /         \_______
      |                  \_____________/
      +-------|---------|------|------|---------|------> radius
           aperture    rim    white blob        rim   aperture
             edge     (FPI boundary)

* The crater floor and steep walls cannot be measured -> a large WHITE blob.
* The FPI boundary is the crest (rim), which lies OUTSIDE the blob. The shaded
  PNG shows it only faintly, so we place it at a fixed fraction of the way
  from the blob's round core to the aperture edge. That fraction was measured
  on the x3p height maps (tools/calibrate_with_x3p.py).

Firing pin drag
---------------
When the barrel unlocks, the pin is still protruding and is dragged across the
primer. This turns the round crater into a "teardrop": a bulge on one side.
We fit a circle to the round part of the blob, treating outward points as
outliers. Whatever sticks out beyond that circle is the drag bulge.
Drag direction = core centre -> centroid of the bulge. No fixed orientation
is assumed anywhere.

The work is split in two because the aperture search (aperture.py) needs the
blob, and the FPI rim needs the aperture:
    detect_blob()  ->  detect_aperture()  ->  detect_firing_pin()
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .geometry import (Circle, disc_mask, fill_holes, fit_circle_lsq,
                       largest_component, morph)
from .primer import Primer


@dataclass
class Blob:
    mask: np.ndarray          # bool: filled no-data blob (crater floor)
    core: Circle              # round part of the blob


@dataclass
class FiringPin:
    blob: np.ndarray
    core: Circle
    rim: Circle                 # FPI boundary (crater rim / crest)
    bulge: np.ndarray           # bool: blob pixels beyond the core (drag)
    drag_detected: bool
    drag_angle_deg: float | None  # 0 = right, 90 = up (counter-clockwise)
    drag_length_px: float | None  # core centre -> tip of the bulge
    drag_tip: tuple | None        # (x, y) of the arrow head
    drag_strength: float          # bulge reach / core radius (0 = round)


def _core_circle(mask: np.ndarray, iters: int = 8) -> Circle:
    """Circle fitted to the ROUND part of a blob's outline.

    Start at the centre of the largest inscribed circle (distance-transform
    maximum, which sits in the round part of a teardrop). Then repeat:
    least-squares fit to the outline points, ignoring points that lie
    clearly OUTSIDE the current circle (the bulge). Inward dents are kept,
    so the fit is not pulled inwards by them either.
    """
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    pts = max(contours, key=len)[:, 0, :].astype(float)
    x, y = pts[:, 0], pts[:, 1]

    dist = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    cy, cx = np.unravel_index(np.argmax(dist), dist.shape)
    d = np.hypot(x - cx, y - cy)
    circle = Circle(float(cx), float(cy), float(np.median(d)))

    for _ in range(iters):
        d = np.hypot(x - circle.cx, y - circle.cy)
        resid = d - circle.r
        mad = np.median(np.abs(resid)) + 1e-6
        keep = resid < 2.0 * mad       # one-sided: drop outward points only
        if keep.sum() < 20:
            break
        circle = fit_circle_lsq(x[keep], y[keep])
    return circle


def detect_blob(gray: np.ndarray, primer: Primer, cfg: Config) -> Blob:
    """Find the large white no-data blob at the bottom of the crater."""
    R = primer.circle.r
    white = ~primer.measured & primer.region
    # An opening wipes out thin white structures (the aperture arc, cracks)
    # so they cannot merge with the blob; the blob is far thicker.
    opened = morph(white, cv2.MORPH_OPEN, cfg.fpi_blob_open * R)
    # Pick the large component closest to the primer centre.
    n, labels, stats, cents = cv2.connectedComponentsWithStats(
        opened.astype(np.uint8), connectivity=8)
    min_area = cfg.fpi_blob_min_area * np.pi * R * R
    best, best_d = None, np.inf
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        d = np.hypot(*(cents[i] - (primer.circle.cx, primer.circle.cy)))
        if d < best_d:
            best, best_d = i, d
    if best is None:
        raise RuntimeError("Firing pin: no central no-data blob found")
    blob = labels == best
    # Smooth: close small notches from drop-outs, then fill holes.
    blob = fill_holes(morph(blob, cv2.MORPH_CLOSE, cfg.fpi_blob_close * R))
    return Blob(blob, _core_circle(blob))


def detect_firing_pin(blob: Blob, aperture_circle: Circle, primer: Primer,
                      cfg: Config) -> FiringPin:
    R = primer.circle.r
    core = blob.core
    shape = blob.mask.shape

    # --- Drag bulge = blob pixels clearly outside the core circle -------
    outside = blob.mask & ~disc_mask(shape, core.scaled(1 + cfg.drag_margin))
    outside = morph(outside, cv2.MORPH_OPEN, 0.02 * R)  # drop thin slivers
    bulge = largest_component(outside)

    drag_detected, angle, length, tip, strength = False, None, None, None, 0.0
    if bulge.any():
        ys, xs = np.nonzero(bulge)
        dist = np.hypot(xs - core.cx, ys - core.cy)
        strength = float((dist.max() - core.r) / core.r)
        area_frac = bulge.sum() / (np.pi * core.r ** 2)
        if area_frac >= cfg.drag_min_area and strength >= cfg.drag_min_reach:
            drag_detected = True
            # Direction: core centre -> bulge centroid (stable against noise)
            dx, dy = xs.mean() - core.cx, ys.mean() - core.cy
            norm = np.hypot(dx, dy)
            angle = float(np.degrees(np.arctan2(-dy, dx)) % 360)
            length = float(dist.max())
            tip = (core.cx + dx / norm * length, core.cy + dy / norm * length)
    if not drag_detected:
        bulge = np.zeros_like(blob.mask)

    # --- Rim of the crater (= FPI boundary) -----------------------------
    rim_r = core.r + cfg.fpi_rim_fraction * (aperture_circle.r - core.r)
    rim = Circle(core.cx, core.cy, rim_r)

    return FiringPin(blob.mask, core, rim, bulge, drag_detected, angle,
                     length, tip, strength)
