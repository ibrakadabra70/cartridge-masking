"""
Tests on SYNTHETIC images, where the right answer is known exactly.

The synthetic "scan" mimics a 3DVM rendering:
* textured grey disc (primer, radius 300 px) on a white background,
* a white firing-pin blob (radius 120 px) with a bulge pointing at 30 deg,
* a thin, BROKEN white ring (aperture edge, radius 211 px).

Run with:   python -m pytest -q
"""

import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import LABELS, Config                    # noqa: E402
from src.geometry import fit_circle_lsq                   # noqa: E402
from src.io_utils import Scan, read_scale_bar, read_x3p   # noqa: E402
from src.pipeline import process_scan                     # noqa: E402

CX, CY = 400.0, 410.0
PRIMER_R, CORE_R, APERTURE_R = 300, 120, 211
DRAG_DEG = 30.0


def synthetic_scan(drag: bool = True, size: int = 820) -> np.ndarray:
    rng = np.random.default_rng(1)
    yy, xx = np.mgrid[:size, :size]
    r = np.hypot(xx - CX, yy - CY)
    ang = np.degrees(np.arctan2(-(yy - CY), xx - CX)) % 360

    img = np.full((size, size), 255, np.uint8)
    img[r <= PRIMER_R] = rng.integers(70, 200, size=(r <= PRIMER_R).sum())

    white = r <= CORE_R
    if drag:  # bulge: a smaller disc pushed out in the drag direction
        bx = CX + 110 * np.cos(np.radians(DRAG_DEG))
        by = CY - 110 * np.sin(np.radians(DRAG_DEG))
        white |= np.hypot(xx - bx, yy - by) <= 60
    # broken aperture arc: 3 px half-width, gaps every 50 degrees
    white |= (np.abs(r - APERTURE_R) <= 3) & ((ang % 50) < 35)
    img[white] = 255
    return img


def make_scan(img, um_per_px=3.0) -> Scan:
    return Scan("synthetic", Path("synthetic.png"), img, um_per_px, "user")


@pytest.fixture(scope="module")
def result():
    return process_scan(make_scan(synthetic_scan()), Config())


def test_primer_centre_and_radius(result):
    c = result.primer.circle
    assert abs(c.cx - CX) < 3 and abs(c.cy - CY) < 3
    assert abs(c.r - PRIMER_R) < 6


def test_aperture_circle_despite_gaps(result):
    a = result.aperture.circle
    assert abs(a.r - APERTURE_R) < 5
    assert np.hypot(a.cx - CX, a.cy - CY) < 5


def test_firing_pin_core_ignores_bulge(result):
    core = result.firing_pin.core
    assert abs(core.r - CORE_R) < 6
    assert np.hypot(core.cx - CX, core.cy - CY) < 6


def test_drag_direction(result):
    fp = result.firing_pin
    assert fp.drag_detected
    err = abs((fp.drag_angle_deg - DRAG_DEG + 180) % 360 - 180)
    assert err < 5


def test_labels_are_consistent(result):
    lab = result.labels
    # centre of the crater is FPI or firing-pin shear, never breech face
    assert lab[int(CY), int(CX)] in (LABELS["firing_pin_impression"],
                                     LABELS["firing_pin_shear"])
    # a point on the breech-face ring (r = 260) is breech face
    assert lab[int(CY), int(CX + 260)] == LABELS["breech_face"]
    # outside the primer is background
    assert lab[5, 5] == LABELS["background"]
    # firing-pin shear lies BEHIND the core centre (opposite the drag)
    ys, xs = np.nonzero(lab == LABELS["firing_pin_shear"])
    d = np.array([np.cos(np.radians(DRAG_DEG)), -np.sin(np.radians(DRAG_DEG))])
    assert ((xs - CX) * d[0] + (ys - CY) * d[1]).mean() < 0


def test_round_blob_has_no_drag():
    res = process_scan(make_scan(synthetic_scan(drag=False)), Config())
    assert not res.firing_pin.drag_detected
    assert not (res.labels == LABELS["firing_pin_shear"]).any()


def test_scale_invariance():
    """Same scan at half resolution -> same sizes in mm."""
    import cv2
    img = synthetic_scan()
    small = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    a = process_scan(make_scan(img, 3.0), Config()).summary()
    b = process_scan(make_scan(small, 6.0), Config()).summary()
    assert abs(a["primer_radius_mm"] - b["primer_radius_mm"]) < 0.02
    assert abs(a["fpi_rim_radius_mm"] - b["fpi_rim_radius_mm"]) < 0.03


def test_fit_circle_exact():
    t = np.linspace(0, 2 * np.pi, 50)
    c = fit_circle_lsq(5 + 7 * np.cos(t), -3 + 7 * np.sin(t))
    assert np.allclose([c.cx, c.cy, c.r], [5, -3, 7], atol=1e-6)


def test_scale_bar_reading():
    """Header with ticks 150 px apart = 0.5 mm -> 3.333 um/px."""
    header = np.full((60, 1200), 255, np.uint8)
    header[20, 600:1060] = 0                      # bar line
    for x in (600, 750, 900, 1050):
        header[8:21, x] = 0                       # ticks
    assert read_scale_bar(header, Config()) == pytest.approx(500 / 150, rel=0.01)


def test_scale_bar_missing():
    assert read_scale_bar(np.full((60, 500), 255, np.uint8), Config()) is None


def test_read_x3p(tmp_path):
    """A tiny hand-made x3p file round-trips through the reader."""
    data = np.array([[1e-6, 2e-6, np.nan], [3e-6, 4e-6, 5e-6]])
    xml = """<p:ISO5436_2><Record1><Axes>
      <CX><AxisType>I</AxisType><DataType>D</DataType><Increment>3.125e-006</Increment></CX>
      <CY><AxisType>I</AxisType><DataType>D</DataType><Increment>3.125e-006</Increment></CY>
      <CZ><AxisType>A</AxisType><DataType>D</DataType><Increment>1</Increment></CZ>
      </Axes></Record1><Record3><MatrixDimension><SizeX>3</SizeX><SizeY>2</SizeY>
      </MatrixDimension><DataLink><PointDataLink>bindata/data.bin</PointDataLink>
      </DataLink></Record3></p:ISO5436_2>"""
    path = tmp_path / "t.x3p"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("main.xml", xml)
        buf = io.BytesIO(data.astype("<f8").tobytes())
        z.writestr("bindata/data.bin", buf.getvalue())
    x = read_x3p(path)
    assert x.heights_um.shape == (2, 3)
    assert x.dx_um == pytest.approx(3.125)
    assert x.heights_um[1, 2] == pytest.approx(5.0)
    assert np.isnan(x.heights_um[0, 2])
