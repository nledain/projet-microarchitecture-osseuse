import numpy as np
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_fill_holes
from skimage.morphology import ball

def fill_3d_holes(volume):
    """
    Fill internal holes inside a 3D binary object.
    """
    filled = np.copy(volume)
    for z in range(volume.shape[0]):
        filled[z] = binary_fill_holes(volume[z])
    return filled

def separate_trabecular_cortical_by_thickness(trinarized_mask, threshold_px=80):
    """
    Separates trabecular and cortical bone based on local thickness (in pixels),
    using label conventions from detour_clean (0=background, 1=dark, 2=bone).
    
    Parameters:
        trinarized_mask (np.ndarray): 3D mask where 2 = bone, 1 = dark background, 0 = outside
        threshold_px (int): thickness threshold in pixels (e.g., 80 for 800 µm if 10 µm/pixel)

    Returns:
        label_mask (np.ndarray): new 3D mask with:
            0 = background
            1 = dark background (unchanged)
            2 = trabecular bone
            3 = cortical bone
    """
    # Step 1: isolate bone
    bone_mask = (trinarized_mask == 2).astype(np.uint8)

    # Step 2: fill 2D holes in each slice to seal cortical shell
    bone_mask_filled = fill_3d_holes(bone_mask)

    # Step 3: compute local thickness
    thickness_map = distance_transform_edt(bone_mask_filled)

    # Step 4: core cortical = thick regions
    core_cortical = (thickness_map > threshold_px/2)

    # Step 5: expand core cortical back outward
    cortical = binary_dilation(core_cortical, structure=ball(threshold_px // 2))

    # Step 6: trabecular = rest of bone not in cortical
    trabecular = bone_mask & (~cortical)

    # Step 7: construct output
    label_mask = np.zeros_like(trinarized_mask, dtype=np.uint8)
    label_mask[trinarized_mask == 1] = 1        # dark inside non-bone
    label_mask[trabecular] = 2                 # trabecular
    label_mask[cortical] = 3                   # cortical

    return label_mask