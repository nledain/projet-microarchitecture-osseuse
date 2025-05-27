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
import cv2

import filtrage_des_images.py as fil
import _test_code_boneanalyzer as bone_an
import 

# Seuils plus stricts
SEUIL_FLOU = 150.0
SEUIL_SOMBRE = 78




chemin_entree = '/home/ttantot/Documents/P2I/projet/E17_10um_1200_1200_1200.tif'
chemin_sortie = '/home/ttantot/Documents/P2I/projet/test_code.tif'
fil.traiter_images_tiff(chemin_entree, chemin_sortie)

tif_path = '/home/ttantot/Documents/P2I/projet/test_code.tif'
img1=Image.open(tif_path)      #pourquoi pas image.open

Frac_tot = 0
cpt=0
for i in range (img1.n_frames):
    img1.seek(i)
    img = np.array(img1)
    
    im = bone_an.detour_clean(img)
    
    Frac_tot += bone_an.bone_fraction(im)
    cpt+=1
print(f'fraction volumique ={Frac_tot/cpt}')