import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, binary_fill_holes, disk
from skimage.feature import canny

def detour_clean(image):
    """Applies edge-based segmentation on a single 2D slice."""
    edges = canny(image, sigma=2.0)
    closed_edges = binary_closing(edges, footprint=disk(3.3))
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
