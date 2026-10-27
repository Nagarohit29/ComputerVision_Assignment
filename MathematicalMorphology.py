import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

IMAGE_PATH = "Turttle-Black-and-White.jpg"
OUTPUT_DIR = Path("morphology_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# -------------------- 1. LOAD AND BINARIZE --------------------
gray = cv.imread(IMAGE_PATH, cv.IMREAD_GRAYSCALE)
if gray is None:
    raise FileNotFoundError(
        f"Cannot load '{IMAGE_PATH}'. Put the image beside this script "
        "or update IMAGE_PATH."
    )

_, binary = cv.threshold(gray, 127, 255, cv.THRESH_BINARY_INV)

kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))

eroded = cv.erode(binary, kernel, iterations=1)
dilated = cv.dilate(binary, kernel, iterations=1)
opened = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=1)
closed = cv.morphologyEx(binary, cv.MORPH_CLOSE, kernel, iterations=1)

SEGMENT_KERNEL_SIZE = 3
segment_kernel = cv.getStructuringElement(
    cv.MORPH_ELLIPSE, (SEGMENT_KERNEL_SIZE, SEGMENT_KERNEL_SIZE)
)
segmentation_mask = cv.morphologyEx(
    binary, cv.MORPH_CLOSE, segment_kernel, iterations=1
)

contours, _ = cv.findContours(
    segmentation_mask.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
)
if not contours:
    raise RuntimeError(
        "No contour found. Check the image and threshold value."
    )

outer_contour = max(contours, key=cv.contourArea)

silhouette = np.zeros_like(binary)
cv.drawContours(
    silhouette, [outer_contour], -1, color=255, thickness=cv.FILLED
)

boundary_kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
silhouette_eroded = cv.erode(silhouette, boundary_kernel, iterations=1)
boundary = cv.subtract(silhouette, silhouette_eroded)

DIRECTION_CODES = {
    (1, 0): 0, (1, -1): 1, (0, -1): 2, (-1, -1): 3,
    (-1, 0): 4, (-1, 1): 5, (0, 1): 6, (1, 1): 7
}

def extract_chain_code(ordered_contour):
    """Convert an ordered 8-connected contour into Freeman direction codes."""
    points = ordered_contour.reshape(-1, 2)
    codes = []

    if len(points) < 2:
        return codes

    for i in range(len(points)):
        current = points[i]
        following = points[(i + 1) % len(points)]

        dx = int(following[0]) - int(current[0])
        dy = int(following[1]) - int(current[1])

        # Normalize step differences to one of the eight directions.
        step = (int(np.sign(dx)), int(np.sign(dy)))
        if step in DIRECTION_CODES:
            codes.append(DIRECTION_CODES[step])

    return codes


chain_code = extract_chain_code(outer_contour)

overlay = cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
cv.drawContours(overlay, [outer_contour], -1, (0, 0, 255), 1)

start_x, start_y = map(int, outer_contour[0, 0])
cv.circle(overlay, (start_x, start_y), 3, (0, 255, 0), -1)

outputs = {
    "input_grayscale.png": gray,
    "binary_silhouette.png": binary,
    "erosion.png": eroded,
    "dilation.png": dilated,
    "opening.png": opened,
    "closing.png": closed,
    "segmentation_mask.png": segmentation_mask,
    "isolated_silhouette.png": silhouette,
    "isolated_boundary.png": boundary,
    "chain_code_contour.png": overlay
}

for filename, result in outputs.items():
    cv.imwrite(str(OUTPUT_DIR / filename), result)

with open(OUTPUT_DIR / "chain_code.txt", "w", encoding="utf-8") as f:
    f.write("8-directional Freeman Chain Code for the segmented outer turtle contour\n")
    f.write(f"Segmentation closing kernel: {SEGMENT_KERNEL_SIZE} x {SEGMENT_KERNEL_SIZE}\n")
    f.write(f"Starting point (x, y): ({start_x}, {start_y})\n")
    f.write(f"Contour points: {len(outer_contour)}\n")
    f.write(f"Chain-code length: {len(chain_code)}\n\n")
    f.write(", ".join(map(str, chain_code)) + "\n")

items = [
    ("Input grayscale", gray),
    ("Binary image", binary),
    ("Erosion", eroded),
    ("Dilation", dilated),
    ("Opening", opened),
    ("Closing", closed),
    ("Segmentation mask (3x3 closing)", segmentation_mask),
    ("Isolated silhouette", silhouette),
    ("Boundary / contour (green=start)", cv.cvtColor(overlay, cv.COLOR_BGR2RGB))
]

fig, axes = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
fig.suptitle("Mathematical Morphology and 8-Directional Chain Code", fontsize=15)

for ax, (title, result) in zip(axes.flat, items):
    if result.ndim == 3:
        ax.imshow(result)
    else:
        ax.imshow(result, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")

fig.savefig(
    OUTPUT_DIR / "morphology_chain_code_comparison.png",
    dpi=160,
    bbox_inches="tight"
)
plt.show()

print("\n--- Mathematical Morphology Results ---")
print("Input dimensions:", gray.shape)
print("Morphology kernel: 3 x 3 rectangular")
print("Segmentation closing kernel:", SEGMENT_KERNEL_SIZE, "x", SEGMENT_KERNEL_SIZE)
print("Contours after segmentation:", len(contours))
print("Selected outer contour area:", cv.contourArea(outer_contour))
print("Contour points:", len(outer_contour))
print("Chain-code starting point (x, y):", (start_x, start_y))
print("Chain-code length:", len(chain_code))
print("First 100 chain-code directions:", chain_code[:100])
print("Saved outputs to:", OUTPUT_DIR.resolve())
print("\nCheck segmentation_mask.png and isolated_silhouette.png.")
print("If the silhouette is still disconnected, try SEGMENT_KERNEL_SIZE = 7.")
print("If important indentations are filled, try SEGMENT_KERNEL_SIZE = 3.")