import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

image = cv.imread("Turtle.jpg")
if image is None:
    raise FileNotFoundError(
        "Turtle.jpg not found. Place it in the same folder as this script "
        "or update the image path."
    )

gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
print("Original image shape:", image.shape)
print("Grayscale image shape:", gray.shape)
print("Image data type:", gray.dtype)

MAX_WIDTH = 1200
if MAX_WIDTH is not None and gray.shape[1] > MAX_WIDTH:
    scale = MAX_WIDTH / gray.shape[1]
    new_size = (MAX_WIDTH, int(gray.shape[0] * scale))
    gray_work = cv.resize(gray, new_size, interpolation=cv.INTER_AREA)
    color_work = cv.resize(image, new_size, interpolation=cv.INTER_AREA)
else:
    gray_work = gray.copy()
    color_work = image.copy()

output_dir = Path("edge_corner_outputs")
output_dir.mkdir(exist_ok=True)

blurred = cv.GaussianBlur(gray_work, (5, 5), 1.0)

lower_threshold = 80
upper_threshold = 160
edges = cv.Canny(blurred, lower_threshold, upper_threshold)

gray_float = np.float32(blurred)
harris_response = cv.cornerHarris(
    gray_float,
    blockSize=2,
    ksize=3,
    k=0.04
)

harris_response_dilated = cv.dilate(harris_response, None)
max_response = harris_response_dilated.max()
corner_threshold = 0.01 * max_response

corner_mask = harris_response_dilated > corner_threshold
corner_overlay = color_work.copy()
corner_overlay[corner_mask] = [0, 0, 255] 

print("Harris maximum response:", float(max_response))
print("Harris threshold:", float(corner_threshold))
print("Pixels marked by threshold:", int(np.count_nonzero(corner_mask)))

cv.imwrite(str(output_dir / "original_turtle.png"), color_work)
cv.imwrite(str(output_dir / "grayscale_turtle.png"), gray_work)
cv.imwrite(str(output_dir / "canny_edges.png"), edges)
cv.imwrite(str(output_dir / "harris_corner_overlay.png"), corner_overlay)

response_min = harris_response.min()
response_max = harris_response.max()
if response_max > response_min:
    response_display = (
        (harris_response - response_min) /
        (response_max - response_min) * 255
    ).astype(np.uint8)
else:
    response_display = np.zeros_like(gray_work)
cv.imwrite(str(output_dir / "harris_response.png"), response_display)

fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
fig.suptitle("Turtle Image: Canny Edge and Harris Corner Detection", fontsize=15)

axes[0].imshow(cv.cvtColor(color_work, cv.COLOR_BGR2RGB))
axes[0].set_title("Original Image")

axes[1].imshow(edges, cmap="gray", vmin=0, vmax=255)
axes[1].set_title("Canny Edge Detection")

axes[2].imshow(cv.cvtColor(corner_overlay, cv.COLOR_BGR2RGB))
axes[2].set_title("Harris Corner Detection")

for ax in axes:
    ax.axis("off")

fig.savefig(output_dir / "edge_corner_comparison.png", dpi=150, bbox_inches="tight")
plt.show()

print("\\nOutput images saved in:", output_dir.resolve())