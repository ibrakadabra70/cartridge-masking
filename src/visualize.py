"""
Drawing the results: colour overlay, side-by-side figure with legend, and a
contact sheet for batch runs.
"""

from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")  # no window needed; we only save files
import matplotlib.pyplot as plt                     # noqa: E402
from matplotlib.lines import Line2D                 # noqa: E402
from matplotlib.patches import Patch                # noqa: E402
import numpy as np                                  # noqa: E402

from .config import LABELS, Config                  # noqa: E402

# Legend text, in the order required by the assignment.
LEGEND = [
    ("breech_face", "Breech-face impression"),
    ("aperture_shear", "Aperture shear"),
    ("firing_pin_shear", "Firing pin shear"),
    ("firing_pin_impression", "Firing pin impression"),
]


def colour_mask(labels: np.ndarray, cfg: Config) -> np.ndarray:
    """RGB image with each label painted in its colour (black = none)."""
    out = np.zeros(labels.shape + (3,), np.uint8)
    for name, _ in LEGEND:
        out[labels == LABELS[name]] = cfg.colours[name]
    return out


def draw_arrow(img: np.ndarray, result, cfg: Config) -> None:
    """Blue drag arrow from the FPI centre to the tip of the drag (in place)."""
    fp = result.firing_pin
    if not fp.drag_detected:
        return
    R = result.primer.circle.r
    start = (int(round(fp.core.cx)), int(round(fp.core.cy)))
    end = (int(round(fp.drag_tip[0])), int(round(fp.drag_tip[1])))
    thick = max(2, int(0.012 * R))
    # White halo first so the arrow stays visible on any background.
    cv2.arrowedLine(img, start, end, (255, 255, 255), thick + 4,
                    cv2.LINE_AA, tipLength=0.18)
    cv2.arrowedLine(img, start, end, cfg.colours["drag_arrow"], thick,
                    cv2.LINE_AA, tipLength=0.18)


def overlay(result, cfg: Config) -> np.ndarray:
    """Scan in grey with semi-transparent coloured regions, outlines and the
    drag arrow (RGB, uint8)."""
    gray = result.scan.gray
    base = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB).astype(np.float32)
    colours = colour_mask(result.labels, cfg).astype(np.float32)
    painted = result.labels > 0
    a = cfg.overlay_alpha
    base[painted] = (1 - a) * base[painted] + a * colours[painted]
    out = base.astype(np.uint8)

    # Thin outline around each region so boundaries are easy to judge.
    thick = max(1, int(0.004 * result.primer.circle.r))
    for name, _ in LEGEND:
        m = (result.labels == LABELS[name]).astype(np.uint8)
        if m.any():
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, contours, -1, cfg.colours[name], thick,
                             cv2.LINE_AA)
    draw_arrow(out, result, cfg)
    return out


def legend_handles(cfg: Config):
    rgb = {k: np.array(v) / 255 for k, v in cfg.colours.items()}
    handles = [Patch(facecolor=rgb[k], edgecolor="k", label=t) for k, t in LEGEND]
    handles.append(Line2D([0], [0], color=rgb["drag_arrow"], lw=3,
                          marker=">", markersize=10, label="Firing pin drag"))
    return handles


def _caption(result) -> str:
    s = result.summary()
    drag = (f"drag {s['drag_angle_deg']:.0f}° (strength {s['drag_strength']:.2f})"
            if s["drag_detected"] else "no drag detected")
    shear = (f"aperture shear {s['aperture_shear_sector_deg']}°"
             if s["aperture_shear_detected"] else "no aperture shear detected")
    return (f"primer radius {s['primer_radius_mm']:.2f} mm  |  FPI rim radius "
            f"{s['fpi_rim_radius_mm']:.2f} mm  |  {drag}  |  {shear}\n"
            f"scale {s['um_per_px']} µm/px ({s['scale_source']})  |  "
            "angles: 0° = right, counter-clockwise")


def side_by_side(result, cfg: Config, path: Path) -> None:
    """Original scan (left) and mask overlay (right) with a colour legend."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 7.6))
    axes[0].imshow(result.scan.gray, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original (3DVM rendering)")
    axes[1].imshow(overlay(result, cfg))
    axes[1].set_title("Generated masks")
    for ax in axes:
        ax.axis("off")
    fig.suptitle(result.scan.name, fontsize=15, fontweight="bold")
    fig.legend(handles=legend_handles(cfg), loc="lower center", ncol=5,
               frameon=False, fontsize=11, bbox_to_anchor=(0.5, 0.06))
    fig.text(0.5, 0.015, _caption(result), ha="center", fontsize=9,
             color="#444")
    if result.warnings:
        fig.text(0.5, 0.905, "warning: " + "; ".join(result.warnings),
                 ha="center", fontsize=9, color="#b00")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.9, bottom=0.13, wspace=0.02)
    fig.savefig(path, dpi=110)
    plt.close(fig)


def contact_sheet(items, cfg: Config, path: Path, cols: int = 6,
                  tile: int = 360) -> None:
    """Grid of small overlays for a quick visual check of a whole batch.
    ``items`` = list of (name, overlay RGB image or None for a failure)."""
    if not items:
        return
    rows = int(np.ceil(len(items) / cols))
    sheet = np.full((rows * tile, cols * tile, 3), 255, np.uint8)
    for i, (name, img) in enumerate(items):
        r, c = divmod(i, cols)
        if img is None:
            t = np.full((tile, tile, 3), 235, np.uint8)
            cv2.putText(t, "FAILED", (tile // 4, tile // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (200, 0, 0), 3)
        else:
            t = cv2.resize(img, (tile, tile), interpolation=cv2.INTER_AREA)
        cv2.putText(t, name, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 0, 0), 2, cv2.LINE_AA)
        sheet[r * tile:(r + 1) * tile, c * tile:(c + 1) * tile] = t
    # Legend strip at the bottom.
    strip = np.full((50, sheet.shape[1], 3), 255, np.uint8)
    x = 10
    for name, text in LEGEND + [("drag_arrow", "Firing pin drag (arrow)")]:
        cv2.rectangle(strip, (x, 15), (x + 25, 38), cfg.colours[name], -1)
        cv2.putText(strip, text, (x + 32, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 0, 0), 1, cv2.LINE_AA)
        x += 50 + 11 * len(text)
    save_rgb(path, np.vstack([sheet, strip]))


def save_rgb(path: Path, rgb: np.ndarray) -> None:
    """Save an RGB image (cv2 expects BGR; imencode handles any path)."""
    ok, buf = cv2.imencode(Path(path).suffix, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise IOError(f"could not encode {path}")
    buf.tofile(str(path))
