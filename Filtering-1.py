import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

image = cv.imread("Turtle.jpg")

if image is None:
    raise FileNotFoundError("Turtle.jpg not found. Check the file path.")

gry_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

print("Original image:", image.shape, image.dtype)
print("Grayscale image:", gry_image.shape, gry_image.dtype)

WORKING_WIDTH = 1200
scale = WORKING_WIDTH / gry_image.shape[1]
working_height = int(gry_image.shape[0] * scale)
clean = cv.resize(gry_image, (WORKING_WIDTH, working_height),
                   interpolation=cv.INTER_AREA)

OUTPUT_DIR = Path("filter_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

def add_salt_pepper_noise(img, amount=0.04, seed=42):
    """Replace random pixels with black or white."""
    rng = np.random.default_rng(seed)
    noisy = img.copy()
    random_values = rng.random(img.shape)

    noisy[random_values < amount / 2] = 0
    noisy[(random_values >= amount / 2) &
          (random_values < amount)] = 255
    return noisy

def add_gaussian_noise(img, sigma=20, seed=42):
    """Add Gaussian-distributed noise and keep values in the 0-255 range."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, sigma, img.shape)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)

def median_filter_custom(img, kernel_size=3):
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number")

    pad = kernel_size // 2
    padded = np.pad(img, pad, mode="edge")
    output = np.empty_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = np.median(region)

    return output

def mean_filter_custom(img, kernel_size=3):
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number")

    pad = kernel_size // 2
    padded = np.pad(img.astype(np.float32), pad, mode="edge")
    output = np.empty_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i + kernel_size, j:j + kernel_size]
            output[i, j] = np.clip(np.mean(region), 0, 255)

    return output

def create_gaussian_kernel(kernel_size=3, sigma=1.0):
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise ValueError("kernel_size must be a positive odd number")
    if sigma <= 0:
        raise ValueError("sigma must be positive")

    radius = kernel_size // 2
    axis = np.arange(-radius, radius + 1)
    x, y = np.meshgrid(axis, axis)

    kernel = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / kernel.sum()

def gaussian_filter_custom(img, kernel_size=3, sigma=1.0):
    kernel = create_gaussian_kernel(kernel_size, sigma)
    pad = kernel_size // 2
    padded = np.pad(img.astype(np.float32), pad, mode="edge")
    output = np.empty_like(img)

    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            region = padded[i:i + kernel_size, j:j + kernel_size]
            value = np.sum(region * kernel)
            output[i, j] = np.clip(value, 0, 255)

    return output

def calculate_mse(reference, result):
    diff = reference.astype(np.float32) - result.astype(np.float32)
    return float(np.mean(diff ** 2))

def calculate_psnr(reference, result):
    mse = calculate_mse(reference, result)
    if mse == 0:
        return float("inf")
    return float(10 * np.log10((255 ** 2) / mse))

salt_noisy = add_salt_pepper_noise(clean, amount=0.04)
gaussian_noisy = add_gaussian_noise(clean, sigma=20)

filters = {
    "Median": lambda x: median_filter_custom(x, 3),
    "Mean": lambda x: mean_filter_custom(x, 3),
    "Gaussian": lambda x: gaussian_filter_custom(x, 3, 1.0),
}

experiments = {
    "Salt-and-Pepper": salt_noisy,
    "Gaussian Noise": gaussian_noisy,
}

cv.imwrite(str(OUTPUT_DIR / "original_grayscale.png"), clean)
cv.imwrite(str(OUTPUT_DIR / "salt_pepper_noisy.png"), salt_noisy)
cv.imwrite(str(OUTPUT_DIR / "gaussian_noisy.png"), gaussian_noisy)

all_results = {}

for noise_name, noisy in experiments.items():
    all_results[noise_name] = {}

    for filter_name, filter_function in filters.items():
        print(f"Applying {filter_name} filter to {noise_name}...")
        filtered = filter_function(noisy)
        all_results[noise_name][filter_name] = filtered

        filename = (
            noise_name.lower().replace("-", "_").replace(" ", "_")
            + "_" + filter_name.lower() + ".png"
        )
        cv.imwrite(str(OUTPUT_DIR / filename), filtered)

for noise_name, noisy in experiments.items():

    images_to_show = [
        ("Original", clean),
        ("Noisy", noisy),
        ("Median Filter", all_results[noise_name]["Median"]),
        ("Mean Filter", all_results[noise_name]["Mean"]),
        ("Gaussian Filter", all_results[noise_name]["Gaussian"])
    ]

    fig, axes = plt.subplots(
        1, 5, figsize=(20, 6),
        constrained_layout=True
    )

    fig.suptitle(
        f"Turtle Image - {noise_name}",
        fontsize=16
    )

    for ax, (title, img) in zip(axes, images_to_show):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=12)
        ax.set_aspect("equal")
        ax.axis("off")

    plt.show()

print("\nMetrics compared against the clean grayscale image")
print(f"{'Noise':<20} {'Filter':<12} {'MSE':>12} {'PSNR (dB)':>12}")

for noise_name, filter_results in all_results.items():
    for filter_name, filtered in filter_results.items():
        mse = calculate_mse(clean, filtered)
        psnr = calculate_psnr(clean, filtered)
        print(f"{noise_name:<20} {filter_name:<12} {mse:>12.2f} {psnr:>12.2f}")

print("\nSaved images and comparison plots to:", OUTPUT_DIR.resolve())
