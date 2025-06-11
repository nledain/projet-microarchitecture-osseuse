import SimpleITK as sitk
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops, label
import skimage
from scipy import scipŷ
import numpy as np
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk
from scipy.ndimage import binary_fill_holes
from skimage.feature import canny
import matplotlib.pyplot as plt

#def edge_detection(image):

edges = canny(image, sigma=2.0)  # Tune sigma based on sharpness

plt.imshow(final_mask, cmap='nipy_spectral')
plt.show()


'''
# Close small gaps in edge to create a full contour
closed_edges = binary_closing(edges, selem=disk(3))

# Fill enclosed area (bone region)
bone_mask = binary_fill_holes(closed_edges)

# Apply threshold only on pixels inside bone
inside_bone_pixels = image[bone_mask]
thresh_inside = threshold_otsu(inside_bone_pixels)

# Create mask of background inside bone
background_inside_bone = (image < thresh_inside) & bone_mask

final_mask = np.zeros_like(image, dtype=np.uint8)
final_mask[bone_mask] = 2                      # Initially mark bone
final_mask[background_inside_bone] = 1         # Overwrite inside-bone background



def detour_clean (img1_array):
    
    edges = canny(img1_array, sigma=2.0)
    closed_edges = binary_closing(edges, footprint=disk(3.3))
    bone_mask = binary_fill_holes(closed_edges)
    inside_bone_pixels = img1_array[bone_mask]
    thresh_inside = threshold_otsu(inside_bone_pixels)
    background_inside_bone = (img1_array < thresh_inside) & bone_mask
    final_mask1 =np.zeros_like(img1_array, dtype=np.uint8)
    final_mask1[bone_mask] = 2                      # Initially mark bone
    final_mask1[background_inside_bone] = 1  
    

    return final_mask1





def read_image(path):
    image = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(image)

def compute_thickness_array(mask):
    dt = distance_transform_edt(mask)
    local_thickness = dt * 2
    return np.mean(local_thickness[local_thickness > 0])

def compute_distance(mask):
    dt = distance_transform_edt(1-mask)
    local_distance = dt * 2
    return np.mean(local_distance[local_distance > 0])

def bone_fraction (img):  #calcul fraction osseuse
    pix_n_sum = np.sum(img==0)   #nbre de pixels noirs
    pix_b_sum = np.sum(img==255)   #nbre de pixels blancs
    frac = pix_b_sum/pix_n_sum    #fraction PAR IMAGE  
    return frac






import SimpleITK as sitk
import numpy as np
import cv2
from skimage.filters import threshold_otsu

def segmentation(volume, type):
    segmented = np.zeros_like(volume, dtype=np.uint8)  # Will store 0, 1, 2

# === Step 2: Loop through slices and classify ===
    for z in range(volume.shape[0]):
        slice_img = volume[z, :, :].astype(np.uint8)

        # Otsu threshold for background vs bone
        t1 = threshold_otsu(slice_img)

        # Mask 1: Bone area (everything inside the specimen)
        bone_mask = slice_img > t1

        # Now within the bone area, re-apply Otsu to separate trabecular vs cortical
        if np.any(bone_mask):
            t2 = threshold_otsu(slice_img[bone_mask])
        else:
            t2 = 255  # prevent crash

    # Assign labels:
    # 0 - background
    # 1 - cortical (intermediate gray values)
    # 2 - trabecular (brightest)

    slice_seg = np.zeros_like(slice_img, dtype=np.uint8)
    slice_seg[slice_img > t1] = 1  # bone mask
    slice_seg[slice_img > t2] = 2  # trabecular overwrite cortical

    segmented[z] = slice_seg

def histogram(image):
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    return hist, cdf_normalized

import matplotlib.pyplot as plt
import numpy as np

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


im = np.array(img)
im_vect = np.ravel(img)
plt.subplot(1,3,1)
plt.imshow(img,cmap='gray')
plt.subplot(1,3,3)
plt.hist(im_vect,bins = 256 ,range = (0,256),color='k')

seuils= ski.filters.threshold_multiotsu(im, classes=3)
print(seuils)
segmented = np.digitize(im, bins=seuils)


if __name__ == "__main__":
'''