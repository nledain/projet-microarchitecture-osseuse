import SimpleITK as sitk
import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops, label

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
