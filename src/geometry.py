"""
Small geometry helpers shared by the detection steps: circles, discs,
angles and morphology kernels.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Circle:
    """A circle in image coordinates (x = column, y = row), in pixels."""
    cx: float
    cy: float
    r: float

    def scaled(self, factor: float) -> "Circle":
        return Circle(self.cx, self.cy, self.r * factor)


def odd(n: float) -> int:
    """Round to the nearest odd integer >= 1 (OpenCV kernels must be odd)."""
    n = max(1, int(round(n)))
    return n if n % 2 == 1 else n + 1


def disk_kernel(diameter_px: float) -> np.ndarray:
    """Elliptical (round) structuring element of the given diameter."""
    d = odd(diameter_px)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d, d))


def disc_mask(shape, circle: Circle) -> np.ndarray:
    """Boolean image that is True inside ``circle``."""
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    return (xx - circle.cx) ** 2 + (yy - circle.cy) ** 2 <= circle.r ** 2


def polar_coords(shape, cx: float, cy: float):
    """Per-pixel radius and angle (radians, 0 = +x/right, counter-clockwise
    as seen on screen, i.e. y axis pointing up) around (cx, cy)."""
    yy, xx = np.mgrid[:shape[0], :shape[1]].astype(np.float32)
    dx, dy = xx - cx, yy - cy
    r = np.hypot(dx, dy)
    theta = np.arctan2(-dy, dx)  # minus: image rows grow downwards
    return r, theta


def unwrap_polar(img: np.ndarray, cx: float, cy: float, r0: float, r1: float,
                 n_theta: int = 720, border: int = 255) -> np.ndarray:
    """Resample ``img`` onto a (radius, angle) grid around (cx, cy).

    Row i = radius r0 + i (1 px steps), column j = angle j * 360/n_theta
    degrees (0 = right, counter-clockwise on screen). In this view, circles
    around the centre become horizontal lines and radial lines become
    vertical lines.
    """
    theta = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    radii = np.arange(r0, r1, 1.0)
    T, Rr = np.meshgrid(theta, radii)
    map_x = (cx + Rr * np.cos(T)).astype(np.float32)
    map_y = (cy - Rr * np.sin(T)).astype(np.float32)
    return cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def angle_diff(a, b):
    """Smallest signed difference between two angles (radians), in [-pi, pi]."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def fit_circle_lsq(x: np.ndarray, y: np.ndarray) -> Circle:
    """Algebraic least-squares circle fit (Kasa method).

    Solves  x^2 + y^2 + D x + E y + F = 0  for D, E, F, which is linear.
    """
    A = np.column_stack([x, y, np.ones_like(x)])
    b = -(x ** 2 + y ** 2)
    (D, E, F), *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy = -D / 2, -E / 2
    r = np.sqrt(max(cx ** 2 + cy ** 2 - F, 0.0))
    return Circle(float(cx), float(cy), float(r))


def circle_from_3_points(p) -> Circle | None:
    """Exact circle through 3 points, or None if they are (nearly) collinear."""
    (x1, y1), (x2, y2), (x3, y3) = p
    d = 2 * (x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    if abs(d) < 1e-6:
        return None
    s1, s2, s3 = x1 ** 2 + y1 ** 2, x2 ** 2 + y2 ** 2, x3 ** 2 + y3 ** 2
    cx = (s1 * (y2 - y3) + s2 * (y3 - y1) + s3 * (y1 - y2)) / d
    cy = (s1 * (x3 - x2) + s2 * (x1 - x3) + s3 * (x2 - x1)) / d
    return Circle(cx, cy, float(np.hypot(x1 - cx, y1 - cy)))


def largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest 8-connected component of a boolean mask."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return np.zeros_like(mask, dtype=bool)
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == biggest


def remove_small(mask: np.ndarray, min_area: float) -> np.ndarray:
    """Drop connected components smaller than ``min_area`` pixels."""
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8)
    keep = np.zeros(n, dtype=bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_area
    return keep[labels]


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill everything enclosed by the outer contours of ``mask``."""
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_NONE)
    out = np.zeros(mask.shape, np.uint8)
    cv2.drawContours(out, contours, -1, 1, thickness=cv2.FILLED)
    return out.astype(bool)


def morph(mask: np.ndarray, op: int, diameter_px: float) -> np.ndarray:
    """Apply a morphological operation with a round kernel; bool in/out."""
    if diameter_px < 1.5:
        return mask.copy()
    out = cv2.morphologyEx(mask.astype(np.uint8), op, disk_kernel(diameter_px))
    return out.astype(bool)
