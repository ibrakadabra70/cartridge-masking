"""
All tunable parameters in one place.

Why "fractions of the primer radius"?
-------------------------------------
Pixel sizes change between scanners, magnifications and calibres. If every
length is written as a fraction of the detected primer radius ``R`` (in
pixels), the same numbers work for a 9 mm scan at 3.3 um/px and for a .40 S&W
scan at 2 um/px. Only a handful of values are true constants (grey-level
thresholds, angles), and they are marked as such.

The default values below were tuned on the NIST Fadul 2011 set (40 scans,
9 mm Luger, Ruger P95). See README.md, section "Methodology", for where each
number comes from.
"""

from dataclasses import dataclass, field


@dataclass
class Config:
    # ------------------------------------------------------------------
    # Input / scale
    # ------------------------------------------------------------------
    # Rows at the top of a 3DVM export that hold the file name + scale bar.
    header_height_px: int = 60
    # Grey level at or above which a pixel is "no data" (the scanner could
    # not measure it, so the renderer left it white).
    no_data_threshold: int = 250
    # Fallback pixel size if the scale bar in the header cannot be read
    # (value printed in the header as "data: 3.12 um/px").
    default_um_per_px: float = 3.12
    # Distance between two neighbouring ticks of the header scale bar.
    scale_bar_tick_mm: float = 0.5
    # A scale-bar reading outside this range is treated as a failure.
    plausible_um_per_px: tuple = (0.5, 20.0)

    # ------------------------------------------------------------------
    # Primer (outer boundary of the measured surface)
    # ------------------------------------------------------------------
    # Primer detection runs BEFORE we know R, so this kernel is a fraction
    # of the image size instead (0.0125 * ~1200 px = 15 px, the value that
    # balanced noise vs. detail in the exploration notebook).
    primer_close_frac_of_image: float = 0.0125

    # ------------------------------------------------------------------
    # Aperture edge (thin white arc between breech face and flow-back)
    # ------------------------------------------------------------------
    # The aperture is searched AROUND the firing-pin blob (found first):
    # radius range as a multiple of the blob's core radius ...
    aperture_r_over_core: tuple = (1.45, 2.1)
    # (measured on the Fadul set: 1.62 - 1.85) ...
    # ... and max. distance between aperture centre and core centre (x R).
    aperture_max_offset: float = 0.12
    # Half-width of the band (fraction of R) used to decide whether a white
    # pixel "lies on" a candidate circle.
    aperture_band: float = 0.012
    # RANSAC iterations for the circle search.
    aperture_ransac_iters: int = 3000
    # Number of angular bins used to measure how much of the circle is
    # supported by white pixels (coverage score).
    aperture_angle_bins: int = 360
    # White blobs smaller than this (fraction of R^2) are ignored as
    # speckle when searching for the arc.
    aperture_min_speckle_area: float = 0.00015

    # ------------------------------------------------------------------
    # Firing pin impression (FPI)
    # ------------------------------------------------------------------
    # The deepest part of the FPI is a large no-data blob. An opening of this
    # size (x R) erases thin white arcs/cracks so they cannot merge with it ...
    fpi_blob_open: float = 0.04
    # ... and a small closing merges ragged drop-outs on its walls into it.
    fpi_blob_close: float = 0.02
    # Blob must be at least this big (fraction of pi*R^2) to be accepted.
    fpi_blob_min_area: float = 0.03
    # Where is the rim of the crater? The shaded PNG does not show it well,
    # so its position was CALIBRATED on the x3p height maps
    # (tools/calibrate_with_x3p.py): rim radius = core radius +
    # fpi_rim_fraction * (aperture radius - core radius).
    fpi_rim_fraction: float = 0.61

    # ------------------------------------------------------------------
    # Firing pin drag (the blob's bulge beyond its round core)
    # ------------------------------------------------------------------
    # A blob pixel counts as "protruding" if it lies further than
    # (1 + drag_margin) * core radius from the core centre.
    drag_margin: float = 0.08
    # Minimum protrusion to call a drag: area (fraction of core disc area)
    # and radial reach beyond the core circle (fraction of core radius).
    drag_min_area: float = 0.015
    drag_min_reach: float = 0.18

    # ------------------------------------------------------------------
    # Aperture shear (polar dash-density analysis, see shear.py)
    # ------------------------------------------------------------------
    # Angular resolution of the polar unwrap (bins over 360 degrees).
    shear_n_theta: int = 720
    # Shear band = FPI rim ... aperture radius + this (x R).
    shear_band_outer: float = 0.03
    # Smoothing of the angular profile S(theta), in degrees.
    shear_smooth_deg: float = 15.0
    # With a known drag, look for the peak within +- this many degrees of
    # the direction opposite to the drag.
    shear_search_half_deg: float = 100.0
    # Peak dash density must be at least this many times the median.
    shear_min_contrast: float = 2.0
    # Sector = contiguous angles where S >= median + level * contrast.
    shear_sector_level: float = 0.5
    shear_min_sector_deg: float = 20.0
    shear_max_sector_deg: float = 150.0

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------
    overlay_alpha: float = 0.45

    # Colours (RGB) required by the assignment.
    colours: dict = field(default_factory=lambda: {
        "breech_face": (220, 30, 30),          # red
        "aperture_shear": (30, 190, 60),       # green
        "firing_pin_shear": (150, 60, 200),    # purple
        "firing_pin_impression": (90, 200, 255),  # light blue
        "drag_arrow": (20, 60, 230),           # blue
    })


# Integer label used for each region in the saved label image.
LABELS = {
    "background": 0,
    "breech_face": 1,
    "aperture_shear": 2,
    "firing_pin_shear": 3,
    "firing_pin_impression": 4,
}
