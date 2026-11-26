"""
Loading scans: 3DVM PNG renderings (primary input) and x3p topography files.

3DVM PNG layout
---------------
    +--------------------------------------------------------------+
    | file path (data: 3.12 um/px)     |----|----|----|   1.59 mm  |  <- header
    |                                                              |     (~60 px)
    |                 shaded rendering of the primer               |
    +--------------------------------------------------------------+

Two different pixel sizes appear in the header:

* ``data: 3.12 um/px`` is the sampling of the raw 3D measurement (x3p).
* The scale bar (ticks every 0.5 mm) gives the pixel size of the PNG itself,
  which was resampled for display (about 3.27 um/px for the numbered Fadul
  scans and 3.14 um/px for the lettered ones).

Measurements made on the PNG must use the second one, so we read the scale bar.
If that fails we fall back to the configured / user-supplied value.
"""

from dataclasses import dataclass
from pathlib import Path
import re
import zipfile

import cv2
import numpy as np

from .config import Config

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass
class Scan:
    """A loaded scan, ready for processing."""
    name: str                 # short name, e.g. "Fadul 1-1"
    path: Path
    gray: np.ndarray          # uint8, header removed
    um_per_px: float          # pixel size of ``gray``
    scale_source: str         # "scale bar", "user", "x3p", or "default"


def short_name(path: Path) -> str:
    """'Fadul 1-1 3DVM.PNG' -> 'Fadul 1-1'."""
    stem = path.stem
    return re.sub(r"\s*3DVM$", "", stem, flags=re.IGNORECASE)


def read_scale_bar(header: np.ndarray, cfg: Config) -> float | None:
    """Estimate the PNG pixel size (um/px) from the tick marks in the header.

    The scale bar is a long horizontal line with short vertical ticks every
    ``cfg.scale_bar_tick_mm``. We find the columns that contain a vertical
    run of dark pixels (the ticks), group neighbouring columns, and convert
    the median tick spacing into um/px.

    Returns None if no sensible scale bar is found.
    """
    if header.size == 0:
        return None
    dark = header < 128
    # The bar line is the row with the most dark pixels.
    bar_row = int(np.argmax(dark.sum(axis=1)))
    if dark[bar_row].sum() < header.shape[1] * 0.15:
        return None  # no long horizontal line -> no scale bar
    # Ticks: columns with a vertical dark run crossing the bar line.
    top, bottom = max(bar_row - 12, 0), bar_row
    tick_cols = np.nonzero(dark[top:bottom].sum(axis=0) >= (bottom - top) * 0.7)[0]
    if tick_cols.size < 3:
        return None
    # Merge adjacent columns (a tick may be 1-2 px thick).
    groups = np.split(tick_cols, np.nonzero(np.diff(tick_cols) > 2)[0] + 1)
    centres = np.array([g.mean() for g in groups])
    if centres.size < 3:
        return None
    spacing = np.median(np.diff(centres))
    um_per_px = cfg.scale_bar_tick_mm * 1000.0 / spacing
    lo, hi = cfg.plausible_um_per_px
    return float(um_per_px) if lo <= um_per_px <= hi else None


def load_scan(path: Path, cfg: Config, um_per_px: float | None = None) -> Scan:
    """Load a 3DVM PNG (or any greyscale image) and remove its header.

    ``um_per_px`` (from the command line) overrides automatic detection.
    """
    path = Path(path)
    # cv2.imread cannot open non-ASCII paths on Windows; imdecode can.
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise IOError(f"Could not read image: {path}")

    h = cfg.header_height_px
    header, gray = img[:h], img[h:]

    if um_per_px is not None:
        scale, source = float(um_per_px), "user"
    else:
        scale = read_scale_bar(header, cfg)
        source = "scale bar"
        if scale is None:
            scale, source = cfg.default_um_per_px, "default"

    return Scan(short_name(path), path, gray, scale, source)


def list_inputs(path: Path) -> list[Path]:
    """A single image, or every 3DVM image in a folder (falls back to every
    image if the folder contains no files named '*3DVM*')."""
    path = Path(path)
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(path)
    images = sorted(p for p in path.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    dvm = [p for p in images if "3dvm" in p.name.lower()]
    return dvm or images


# ----------------------------------------------------------------------
# x3p (ISO 25178-72) reader
# ----------------------------------------------------------------------
@dataclass
class X3P:
    heights_um: np.ndarray    # float, NaN where not measured; [row, col]
    dx_um: float              # sample spacing along x
    dy_um: float              # sample spacing along y


def read_x3p(path: Path) -> X3P:
    """Minimal x3p reader (no extra dependency).

    An x3p file is a zip archive with ``main.xml`` (metadata) and a binary
    matrix (``bindata/data.bin``). The NIST files store float64 heights in
    metres, row-major [y, x], with NaN for missing points. Only that layout
    is supported here; anything else raises an error.
    """
    with zipfile.ZipFile(path) as z:
        xml = z.read("main.xml").decode("utf-8", "replace")

        def tag(name, text=xml):
            m = re.search(rf"<{name}>\s*([^<]+?)\s*</{name}>", text)
            if not m:
                raise ValueError(f"x3p: <{name}> not found in {path}")
            return m.group(1)

        nx, ny = int(tag("SizeX")), int(tag("SizeY"))
        cx = re.search(r"<CX>(.*?)</CX>", xml, re.S).group(1)
        cy = re.search(r"<CY>(.*?)</CY>", xml, re.S).group(1)
        cz = re.search(r"<CZ>(.*?)</CZ>", xml, re.S).group(1)
        dx, dy = float(tag("Increment", cx)), float(tag("Increment", cy))
        dtype = {"D": "<f8", "F": "<f4", "L": "<i4", "I": "<i2"}[tag("DataType", cz)]
        zscale = float(tag("Increment", cz))
        link = tag("PointDataLink")
        raw = np.frombuffer(z.read(link), dtype=dtype)

    if raw.size != nx * ny:
        raise ValueError(f"x3p: expected {nx * ny} points, found {raw.size}")
    heights = raw.reshape(ny, nx).astype(np.float64) * zscale * 1e6  # -> um
    return X3P(heights, dx * 1e6, dy * 1e6)


def x3p_as_gray(x3p: X3P) -> np.ndarray:
    """Turn an x3p height map into a 3DVM-like image for the same pipeline:
    measured points -> mid grey, missing points -> white (255)."""
    gray = np.full(x3p.heights_um.shape, 255, np.uint8)
    gray[~np.isnan(x3p.heights_um)] = 128
    return gray
