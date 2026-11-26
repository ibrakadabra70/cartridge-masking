"""Explore a single cartridge case scan."""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

# Project root = the folder above src/, no matter where the script is run from
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "fadul_2011"
image_path = DATA_DIR / "Fadul 1-1 3DVM.PNG"

# --- Step 1: load the image as greyscale ---
image = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError(f"Could not read {image_path}")

print("Shape (height, width):", image.shape)

# --- Step 2: remove the text header ---
HEADER_HEIGHT = 60  # rows of text + scale bar at the top of 3DVM exports
cropped = image[HEADER_HEIGHT:, :]

# --- Step 3: separate measured surface from "no data" ---
WHITE_THRESHOLD = 250  # pixels brighter than this count as background
data_mask = cropped < WHITE_THRESHOLD

# --- Step 4: clean up the mask ---
mask_u8 = data_mask.astype(np.uint8) * 255
KERNEL_SIZE = 15
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KERNEL_SIZE, KERNEL_SIZE))
opened = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
cleaned = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

# --- Step 5: keep only the biggest white blob (drops the corners) ---
num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(cleaned)
areas = stats[1:, cv2.CC_STAT_AREA]
largest_label = 1 + np.argmax(areas)
primer_mask = labels == largest_label

# --- Step 6: fill the primer into a solid disc ---
# Trace the outer outline of the blob
contours, _ = cv2.findContours(
    primer_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
)
outer_contour = max(contours, key=cv2.contourArea)

# Paint everything inside that outline white
solid_primer = np.zeros_like(mask_u8)
cv2.drawContours(solid_primer, [outer_contour], -1, color=255, thickness=cv2.FILLED)

# --- Step 7: describe the disc as a circle (centre + radius) ---
moments = cv2.moments(solid_primer, binaryImage=True)
area = moments["m00"]                 # number of white pixels
cx = moments["m10"] / area            # average x of white pixels
cy = moments["m01"] / area            # average y of white pixels
radius = np.sqrt(area / np.pi)        # radius of a circle with the same area

MICRONS_PER_PIXEL = 3.12  # from the header text on the scan
radius_mm = radius * MICRONS_PER_PIXEL / 1000
print(f"Primer centre: ({cx:.0f}, {cy:.0f}) px")
print(f"Primer radius: {radius:.0f} px = {radius_mm:.2f} mm "
      f"(diameter {2 * radius_mm:.2f} mm)")

# --- Draw the circle on top of the scan ---
overlay = cv2.cvtColor(cropped, cv2.COLOR_GRAY2RGB)
centre = (round(cx), round(cy))
cv2.circle(overlay, centre, round(radius), color=(255, 0, 0), thickness=4)
cv2.circle(overlay, centre, 8, color=(255, 0, 0), thickness=-1)

# --- Show the results ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
stages = [
    (primer_mask, "1. Largest blob"),
    (solid_primer, "2. Filled disc"),
    (overlay, "3. Fitted circle on scan"),
]
for ax, (img, title) in zip(axes, stages):
    ax.imshow(img, cmap="gray")
    ax.set_title(title)
    ax.axis("off")

plt.tight_layout()
plt.show()