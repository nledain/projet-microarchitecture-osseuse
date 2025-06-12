import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk
from scipy.ndimage import binary_fill_holes
from skimage.feature import canny
import matplotlib.pyplot as plt
import tifffile

def detour_clean(image):
    """Applies edge-based segmentation on a single 2D slice."""
    edges = canny(image, sigma=2.0)
    closed_edges = binary_closing(edges, footprint=disk(5))
    bone_mask = binary_fill_holes(closed_edges)

    inside_bone_pixels = image[bone_mask]
    if inside_bone_pixels.size == 0:
        return np.zeros_like(image, dtype=np.uint8)

    thresh_inside = threshold_otsu(inside_bone_pixels)
    background_inside_bone = (image < thresh_inside) & bone_mask

    final_mask = np.zeros_like(image, dtype=np.uint8)
    final_mask[bone_mask] = 2
    final_mask[background_inside_bone] = 1

    return final_mask

def detour_clean_stack(volume):
    """
    Apply detour_clean to an entire 3D volume slice-by-slice.

    Parameters:
        volume (np.ndarray): 3D numpy array of shape (N_slices, H, W)

    Returns:
        np.ndarray: 3D labeled mask of same shape as volume (0: background, 1: inside-bone dark, 2: bone)
    """
    mask_stack = np.zeros_like(volume, dtype=np.uint8)
    for z in range(volume.shape[0]):
        mask_stack[z] = detour_clean(volume[z])
    return mask_stack


if __name__ == "__main__":
    stack = tifffile.imread(r"c:\Users\20109\Desktop\P2I8 - Projet\P2I8 Data\E10_10um_1200_1200_1200.tif")
    image = stack[500]
    mask = detour_clean(image)
    plt.imshow(mask, cmap='gray')
    plt.title("Masque segmenté (slice 500)")
    plt.axis("off")
    plt.show()

