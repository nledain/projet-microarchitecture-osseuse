import matplotlib.pyplot as plt

def histogram(image, title = f"Histogram of {image}", bins=256):
    """
    Plots the histogram of a 2D grayscale image.

    Parameters:
        image (np.ndarray): Grayscale image (2D array).
        title (str): Title of the plot.
        bins (int): Number of bins to use in the histogram.
    """
    if image.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array.")

    plt.figure()
    plt.hist(image.ravel(), bins=bins, range=(0, 255), color='black')
    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
