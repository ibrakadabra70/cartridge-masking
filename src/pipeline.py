"""
The full pipeline for one scan, and the result object it returns.

Order of the steps (each one uses the results of the previous ones):

    load PNG ─► primer disc ─► firing-pin blob ─► aperture circle
                     │               │                  │
                     ▼               ▼                  ▼
               breech face     FPI + drag        aperture shear
                  (red)     (light blue, arrow,    (green)
                             purple)

Overlaps are resolved so that every pixel gets ONE label. Priority (highest
first): firing pin shear > firing pin impression > aperture shear >
breech face. That way the small, specific regions are never hidden by the
large ones.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path

import cv2
import numpy as np

from .aperture import Aperture, detect_aperture
from .config import LABELS, Config
from .firing_pin import FiringPin, detect_blob, detect_firing_pin
from .geometry import Circle, disc_mask, morph
from .io_utils import Scan, load_scan, short_name
from .primer import Primer, detect_primer
from .shear import (ApertureShear, detect_aperture_shear,
                    detect_firing_pin_shear)
from .topography import rim_fraction_from_x3p


@dataclass
class Result:
    scan: Scan
    primer: Primer
    aperture: Aperture
    firing_pin: FiringPin
    aperture_shear: ApertureShear
    labels: np.ndarray                       # uint8 label image (see LABELS)
    rim_fraction: float                      # used for the FPI rim
    rim_source: str                          # "default" or "x3p"
    warnings: list = field(default_factory=list)

    def mask(self, name: str) -> np.ndarray:
        return self.labels == LABELS[name]

    # ---- numbers for the summary CSV ---------------------------------
    def summary(self) -> dict:
        s = self.scan.um_per_px / 1000.0          # mm per px
        p, ap, fp = self.primer.circle, self.aperture.circle, self.firing_pin
        area_mm2 = lambda m: round(float(m.sum()) * s * s, 4)  # noqa: E731
        return {
            "file": self.scan.path.name,
            "status": "ok",
            "um_per_px": round(self.scan.um_per_px, 3),
            "scale_source": self.scan.scale_source,
            "primer_cx_px": round(p.cx, 1),
            "primer_cy_px": round(p.cy, 1),
            "primer_radius_px": round(p.r, 1),
            "primer_radius_mm": round(p.r * s, 3),
            "aperture_radius_mm": round(ap.r * s, 3),
            "aperture_arc_coverage": round(self.aperture.coverage, 2),
            "fpi_cx_px": round(fp.core.cx, 1),
            "fpi_cy_px": round(fp.core.cy, 1),
            "fpi_rim_radius_mm": round(fp.rim.r * s, 3),
            "fpi_rim_fraction": round(self.rim_fraction, 3),
            "fpi_rim_source": self.rim_source,
            "breech_face_area_mm2": area_mm2(self.mask("breech_face")),
            "fpi_area_mm2": area_mm2(self.mask("firing_pin_impression")),
            "aperture_shear_area_mm2": area_mm2(self.mask("aperture_shear")),
            "firing_pin_shear_area_mm2": area_mm2(self.mask("firing_pin_shear")),
            "drag_detected": fp.drag_detected,
            "drag_angle_deg": None if fp.drag_angle_deg is None
            else round(fp.drag_angle_deg, 1),
            "drag_strength": round(fp.drag_strength, 2),
            "drag_length_mm": None if fp.drag_length_px is None
            else round(fp.drag_length_px * s, 3),
            "aperture_shear_detected": self.aperture_shear.detected,
            "aperture_shear_sector_deg": None
            if self.aperture_shear.sector_deg is None
            else "{:.0f}-{:.0f}".format(*self.aperture_shear.sector_deg),
            "aperture_shear_contrast": round(self.aperture_shear.contrast, 2),
            "warnings": "; ".join(self.warnings),
        }


def build_labels(primer: Primer, aperture: Aperture, fp: FiringPin,
                 ap_shear: ApertureShear, fp_shear: np.ndarray,
                 cfg: Config) -> np.ndarray:
    shape = primer.measured.shape
    R = primer.circle.r
    ap = aperture.circle

    # Breech face: the primer outside the aperture (with a small margin so
    # the white arc itself is not included). Small no-data specks are
    # closed, large voids stay out.
    measured = morph(primer.measured, cv2.MORPH_CLOSE, 0.03 * R)
    breech = (primer.region & measured &
              ~disc_mask(shape, Circle(ap.cx, ap.cy, ap.r + 0.02 * R)))

    # Firing pin impression: crater up to the rim (+ any part of the blob
    # outside it), never beyond the aperture.
    fpi = (disc_mask(shape, fp.rim) | fp.blob) & disc_mask(shape, ap)

    labels = np.zeros(shape, np.uint8)
    # Paint from lowest to highest priority.
    labels[breech] = LABELS["breech_face"]
    labels[ap_shear.mask] = LABELS["aperture_shear"]
    labels[fpi] = LABELS["firing_pin_impression"]
    labels[fp_shear] = LABELS["firing_pin_shear"]
    return labels


def process(path: Path, cfg: Config, um_per_px: float | None = None,
            use_x3p: bool = False) -> Result:
    """Run every step on one image file and return a Result."""
    scan = load_scan(path, cfg, um_per_px)
    x3p_path = _matching_x3p(Path(path)) if use_x3p else None
    return process_scan(scan, cfg, use_x3p=use_x3p, x3p_path=x3p_path)


def process_scan(scan: Scan, cfg: Config, use_x3p: bool = False,
                 x3p_path: Path | None = None) -> Result:
    """Run every step on an already-loaded scan (header removed)."""
    gray = scan.gray
    warnings = []
    if scan.scale_source == "default":
        warnings.append("scale bar not found; default um/px used")

    primer = detect_primer(gray, cfg)
    if primer.clipped:
        warnings.append("primer touches image border")
    blob = detect_blob(gray, primer, cfg)
    aperture = detect_aperture(primer, blob, cfg)
    if aperture.coverage < 0.5:
        warnings.append(f"aperture arc weak (coverage {aperture.coverage:.2f})")

    # Optional advanced mode: measure this scan's own rim position from 3D.
    rim_fraction, rim_source = cfg.fpi_rim_fraction, "default"
    if use_x3p:
        if x3p_path is None:
            warnings.append("--use-x3p: no matching .x3p file")
        else:
            try:
                rim_fraction = rim_fraction_from_x3p(x3p_path, cfg)
                rim_source = "x3p"
            except Exception as exc:  # keep going with the default
                warnings.append(f"x3p rim estimate failed: {exc}")
    run_cfg = replace(cfg, fpi_rim_fraction=rim_fraction)

    fp = detect_firing_pin(blob, aperture.circle, primer, run_cfg)
    ap_shear = detect_aperture_shear(gray, primer, aperture, fp, run_cfg)
    fp_shear = detect_firing_pin_shear(gray.shape, aperture, fp, run_cfg)
    labels = build_labels(primer, aperture, fp, ap_shear, fp_shear, run_cfg)
    return Result(scan, primer, aperture, fp, ap_shear, labels,
                  rim_fraction, rim_source, warnings)


def _matching_x3p(png_path: Path) -> Path | None:
    """'Fadul 1-1 3DVM.PNG' -> 'Fadul 1-1.x3p' in the same folder."""
    cand = png_path.with_name(short_name(png_path) + ".x3p")
    return cand if cand.exists() else None

