def detour_clean (img1_array):
    
    edges = canny(img1_array, sigma=2.0)
    closed_edges = binary_closing(edges, footprint=disk(3.3))
    bone_mask = binary_fill_holes(closed_edges)
    inside_bone_pixels = img1_array[bone_mask]
    thresh_inside = threshold_otsu(inside_bone_pixels)
    background_inside_bone = (img1_array < thresh_inside) & bone_mask
    final_mask =np.zeros_like(img1_array, dtype=np.uint8)
    final_mask[bone_mask] = 2                      # Initially mark bone
    final_mask[background_inside_bone] = 1  

    return final_mask