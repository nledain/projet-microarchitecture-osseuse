from skimage import io
from skimage.color import rgb2gray
import SimpleITK as sitk
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops, label
import skimage
from skimage.filters import threshold_otsu
from skimage.morphology import binary_closing, disk
from scipy.ndimage import binary_fill_holes
from skimage.feature import canny
import matplotlib.pyplot as plt


def compute_thickness(volume):
    # Set everything except 2 to 0
    volume = np.where(volume == 2, 1, 0)
    dt = distance_transform_edt(volume)
    local_thickness = dt * 2
    return np.mean(local_thickness[local_thickness > 0])

def compute_spacing(volume):
    # Set everything except 1 to 0
    volume = np.where(volume == 1, 1, 0)
    dt = distance_transform_edt(volume)
    local_distance = dt * 2
    return np.mean(local_distance[local_distance > 0])


def bone_fraction (img):  #calcul fraction osseuse
    pix_n_sum = np.sum(img==1)   #nbre de pixels noirs
    pix_b_sum = np.sum(img==2)   #nbre de pixels blancs
    frac = pix_b_sum/(pix_n_sum + pix_b_sum)   #fraction PAR IMAGE  
    return frac


tif_path = '/home/ttantot/Documents/P2I/projet/seance 2/Capture d’écran du 2025-05-21 15-54-27.png'
img1=io.imread(tif_path)
img= np.array(img1)
gray_img = rgb2gray(img1)
# img.seek(706)
#def edge_detection(image):

edges = canny(gray_img, sigma=2.0)  # Tune sigma based on sharpness

# Close small gaps in edge to create a full contour
closed_edges = binary_closing(edges, selem=disk(3))

# Fill enclosed area (bone region)
bone_mask = binary_fill_holes(closed_edges)

# Apply threshold only on pixels inside bone
inside_bone_pixels = gray_img[bone_mask]
thresh_inside = threshold_otsu(inside_bone_pixels)

# Create mask of background inside bone
background_inside_bone = (gray_img < thresh_inside) & bone_mask

final_mask = np.zeros_like(gray_img, dtype=np.uint8)
final_mask[bone_mask] = 2                      # Initially mark bone
final_mask[background_inside_bone] = 1         # Overwrite inside-bone background

print(compute_thickness_array(final_mask))
print(compute_distance(final_mask))
print(bone_fraction(final_mask))

plt.imshow(edges, cmap='nipy_spectral')
plt.show()




