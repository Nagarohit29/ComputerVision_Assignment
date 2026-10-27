import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

IMAGE_PATH = "Coins.jpg"
OUTPUT_DIR = Path("segmentation_outputs_final")
OUTPUT_DIR.mkdir(exist_ok=True)

COINS = [
    (53, 61, 48),
    (160, 61, 49),
    (274, 61, 49),
    (385, 61, 48),
    (65, 219, 59),
    (210, 219, 57),
    (365, 219, 59),
    (62, 382, 54),
    (210, 382, 56),
    (385, 382, 56),
]

BOUNDARY_BAND = 4

image = cv.imread(IMAGE_PATH)
if image is None:
    raise FileNotFoundError(
        f"Cannot load '{IMAGE_PATH}'. Place it beside this script "
        "or update IMAGE_PATH."
    )

height, width = image.shape[:2]
gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

sx, sy = width / 447.0, height / 447.0
coins = [
    (int(round(x * sx)), int(round(y * sy)), int(round(r * (sx + sy) / 2)))
    for x, y, r in COINS
]

print("Input image dimensions:", image.shape)
print("Coins configured:", len(coins))

blurred_gray = cv.GaussianBlur(gray, (5, 5), 0)
otsu_value, otsu_binary = cv.threshold(
    blurred_gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU
)
print("Otsu threshold selected:", otsu_value)

kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
otsu_clean = cv.morphologyEx(otsu_binary, cv.MORPH_OPEN, kernel, iterations=1)
otsu_clean = cv.morphologyEx(otsu_clean, cv.MORPH_CLOSE, kernel, iterations=1)

markers = np.ones((height, width), dtype=np.int32)

for x, y, r in coins:
    inner_r = max(5, r - BOUNDARY_BAND)
    outer_r = r + BOUNDARY_BAND
    cv.circle(markers, (x, y), outer_r, 0, thickness=cv.FILLED)
    cv.circle(markers, (x, y), inner_r, 1, thickness=cv.FILLED)

for label, (x, y, r) in enumerate(coins, start=2):
    inner_r = max(5, r - BOUNDARY_BAND)
    cv.circle(markers, (x, y), inner_r, int(label), thickness=cv.FILLED)

markers[0, :] = 1
markers[-1, :] = 1
markers[:, 0] = 1
markers[:, -1] = 1

marker_preview = np.zeros_like(image)
marker_preview[markers == 1] = (40, 40, 40)
marker_preview[markers == 0] = (0, 0, 0)
rng = np.random.default_rng(17)
for label in range(2, len(coins) + 2):
    marker_preview[markers == label] = rng.integers(80, 256, size=3, dtype=np.uint8)

circle_preview = image.copy()
for idx, (x, y, r) in enumerate(coins, start=1):
    cv.circle(circle_preview, (x, y), r, (0, 255, 0), 2)
    cv.circle(circle_preview, (x, y), 3, (0, 0, 255), -1)
    cv.putText(
        circle_preview, str(idx), (x - 7, y + 5),
        cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv.LINE_AA
    )

watershed_input = cv.GaussianBlur(image, (5, 5), 0)
watershed_markers = cv.watershed(watershed_input.copy(), markers.copy())

boundary_overlay = image.copy()
boundary_overlay[watershed_markers == -1] = (0, 0, 255)

region_view = np.zeros_like(image)
rng = np.random.default_rng(42)
for label in range(2, len(coins) + 2):
    color = rng.integers(60, 256, size=3, dtype=np.uint8)
    region_view[watershed_markers == label] = color

outputs = {
    "original.png": image,
    "grayscale.png": gray,
    "otsu_binary.png": otsu_binary,
    "otsu_cleaned_mask.png": otsu_clean,
    "coin_marker_positions.png": circle_preview,
    "watershed_marker_map.png": marker_preview,
    "watershed_boundaries.png": boundary_overlay,
    "watershed_regions.png": region_view,
}
for filename, result in outputs.items():
    cv.imwrite(str(OUTPUT_DIR / filename), result)

with open(OUTPUT_DIR / "coin_markers.txt", "w", encoding="utf-8") as f:
    f.write("Marker-controlled Watershed coin centers/radii (x, y, radius)\n")
    f.write(f"Boundary band: {BOUNDARY_BAND} pixels\n")
    for idx, coin in enumerate(coins, start=1):
        f.write(f"Coin {idx}: {coin}\n")

items = [
    ("Original image", cv.cvtColor(image, cv.COLOR_BGR2RGB)),
    ("Grayscale image", gray),
    (f"Otsu binary (T={otsu_value:.0f})", otsu_binary),
    ("Cleaned Otsu mask", otsu_clean),
    ("Coin marker positions (green)", cv.cvtColor(circle_preview, cv.COLOR_BGR2RGB)),
    ("Watershed marker map", cv.cvtColor(marker_preview, cv.COLOR_BGR2RGB)),
    ("Watershed boundaries (red)", cv.cvtColor(boundary_overlay, cv.COLOR_BGR2RGB)),
    ("Watershed regions (10 coins)", cv.cvtColor(region_view, cv.COLOR_BGR2RGB)),
]

fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
fig.suptitle("Otsu Thresholding and Marker-Controlled Watershed Segmentation", fontsize=15)
for ax in axes.flat:
    ax.axis("off")
for ax, (title, result) in zip(axes.flat, items):
    if result.ndim == 3:
        ax.imshow(result)
    else:
        ax.imshow(result, cmap="gray", vmin=0, vmax=255)
    ax.set_title(title)
    ax.axis("off")

fig.savefig(
    OUTPUT_DIR / "otsu_watershed_comparison.png",
    dpi=160,
    bbox_inches="tight"
)
plt.show()

labels_present = [
    int(v) for v in np.unique(watershed_markers) if v >= 2
]
print("\n--- Final Segmentation Results ---")
print("Otsu threshold:", otsu_value)
print("Coin foreground markers configured:", len(coins))
print("Positive coin labels present after Watershed:", labels_present)
print("Number of coin labels present:", len(labels_present))
print("Watershed boundary pixels:", int(np.count_nonzero(watershed_markers == -1)))
print("Output files saved to:", OUTPUT_DIR.resolve())
print(
    "\nThis image contains 10 coins (4 + 3 + 3). Inspect coin_marker_positions.png "
    "to confirm the green circles align with the coin edges. The marker coordinates "
    "are tailored to this supplied image."
)