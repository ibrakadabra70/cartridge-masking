"""
Step 1: find the primer (the whole measured disc) and its centre / radius.

Method (extends notebooks/01_explore_primer.py):

1. "Measured surface" = pixels darker than the no-data threshold.
2. Morphological opening + closing with a medium kernel (~15 px) removes
   speckle and bridges small gaps along the ragged outer edge.
3. Keep the largest connected component. This drops the small scan fragments
   in the four image corners.
4. Fill its outer contour -> a solid region (this also fills the white
   firing-pin blob and the aperture arc in the middle).
5. Centre + radius:
   * Normal case: centroid of the region (image moments) and
     radius = sqrt(area / pi). Robust, because it averages every pixel.
   * If the disc touches the image border (a few scans are off-centre and
     clipped), the area is too small and the centroid is pulled inwards.
     Then we fit a circle by least squares to the outline points that are
     NOT on the image border instead.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .config import Config
from .geometry import Circle, fill_holes, fit_circle_lsq, largest_component, morph


@dataclass
class Primer:
    circle: Circle            # centre + radius R
    measured: np.ndarray      # bool: pixel holds real data (< threshold)
    region: np.ndarray        # bool: filled primer region (outer boundary)
    clipped: bool             # True if the primer touches the image border


def measured_mask(gray: np.ndarray, cfg: Config) -> np.ndarray:
    """True where the scanner returned data (not pure white)."""
    return gray < cfg.no_data_threshold


def detect_primer(gray: np.ndarray, cfg: Config) -> Primer:
    measured = measured_mask(gray, cfg)

    k = cfg.primer_close_frac_of_image * min(gray.shape)
    cleaned = morph(measured, cv2.MORPH_OPEN, k)
    cleaned = morph(cleaned, cv2.MORPH_CLOSE, k)

    main = largest_component(cleaned)
    if not main.any():
        raise RuntimeError("No measured surface found (image all white?)")
    region = fill_holes(main)

    m = cv2.moments(region.astype(np.uint8), binaryImage=True)
    circle = Circle(m["m10"] / m["m00"], m["m01"] / m["m00"],
                    float(np.sqrt(m["m00"] / np.pi)))

    # Outline points away from the image border.
    contours, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    pts = max(contours, key=len)[:, 0, :].astype(float)
    h, w = gray.shape
    margin = 3
    inside = ((pts[:, 0] > margin) & (pts[:, 0] < w - 1 - margin) &
              (pts[:, 1] > margin) & (pts[:, 1] < h - 1 - margin))
    clipped = inside.mean() < 0.97
    if clipped and inside.sum() > 50:
        circle = fit_circle_lsq(pts[inside, 0], pts[inside, 1])

    return Primer(circle, measured, region, bool(clipped))
