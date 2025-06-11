import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import threshold_otsu, laplace
from tifffile import TiffFile


# Plotting blurriness and brightness and computing Otsu thresholds
def plot_otsu_histograms(image_stack):
    blur_scores = []
    brightness_scores = []

    for img in image_stack:
        #img = img.astype(np.float32)
        #img = (img - img.min()) / (img.ptp() + 1e-8)  # Normalize to [0,1]
        
        blur_scores.append(laplace(img).var())
        brightness_scores.append(np.mean(img))

    # Convert to arrays
    blur_scores = np.array(blur_scores)
    brightness_scores = np.array(brightness_scores)

    # Compute Otsu thresholds
    blur_thresh = threshold_otsu(blur_scores)
    brightness_thresh = threshold_otsu(brightness_scores)

    # Plot Laplacian variance (blur measure)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(blur_scores, bins=50, color='skyblue', edgecolor='black')
    plt.axvline(blur_thresh, color='red', linestyle='--', label=f'Otsu: {blur_thresh:.4f}')
    plt.title("Laplacian Variance (Blurriness)")
    plt.xlabel("Laplacian Variance")
    plt.ylabel("Slice Count")
    plt.legend()

    # Plot Mean Brightness
    plt.subplot(1, 2, 2)
    plt.hist(brightness_scores, bins=50, color='lightgreen', edgecolor='black')
    plt.axvline(brightness_thresh, color='red', linestyle='--', label=f'Otsu: {brightness_thresh:.4f}')
    plt.title("Mean Brightness per Slice")
    plt.xlabel("Mean Intensity")
    plt.ylabel("Slice Count")
    plt.legend()

    plt.tight_layout()
    plt.show()

    return blur_thresh, brightness_thresh


# Function to plot stack metrics: mean brightness and Laplacian variance (as a function of image position in the stack)
def plot_stack_metrics(tiff_path):
    # Load stack
    with TiffFile(tiff_path) as tif:
        stack = tif.asarray()

    # Compute mean brightness and Laplacian variance
    brightness = [img.mean() for img in stack]
    lap_var = [laplace(img).var() for img in stack]

    indices = np.arange(1, len(stack) + 1)  # From 1 to N

    # Plot
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(indices, brightness, color='green')
    plt.title("Mean Brightness Across Stack")
    plt.xlabel("Slice Index")
    plt.ylabel("Mean Intensity (0–255)")
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.subplot(1, 2, 2)
    plt.plot(indices, lap_var, color='blue')
    plt.title("Blurriness (Laplacian Variance) Across Stack")
    plt.xlabel("Slice Index")
    plt.ylabel("Laplacian Variance")
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.show()

# === Replace with your file ===
tiff_path = r"C:\Users\20109\Desktop\P2I8 - Projet\P2I8 Data\E17_10um_1200_1200_1200.tif"

plot_stack_metrics(tiff_path)

with TiffFile(tiff_path) as tif:
    stack = tif.asarray()

blur_t, bright_t = plot_otsu_histograms(stack)
print(f"Otsu blur threshold: {blur_t}")
print(f"Otsu brightness threshold: {bright_t}")