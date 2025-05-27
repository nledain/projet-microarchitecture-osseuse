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



# Seuils plus stricts
SEUIL_FLOU = 150.0
SEUIL_SOMBRE = 78

def est_floue(image_np, seuil=SEUIL_FLOU):
    variance = cv2.Laplacian(image_np, cv2.CV_64F).var()
    return variance < seuil

def est_trop_sombre(image_np, seuil=SEUIL_SOMBRE):
    moyenne = np.mean(image_np)
    return moyenne < seuil

def ameliorer_contraste(image_np):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image_np)

def traiter_images_tiff(chemin_tiff_entree, chemin_tiff_sortie):
    image_tiff = Image.open(chemin_tiff_entree)
    images_conservees = []
    total_tranches = image_tiff.n_frames

    for indice in range(total_tranches):
        image_tiff.seek(indice)
        image_gris = np.array(image_tiff.convert('L'))

        # On teste l'image brute AVANT traitement
        if est_floue(image_gris) or est_trop_sombre(image_gris):
            continue  # on saute cette image

        # Si ok → on améliore l’image pour l’export
        image_amelioree = ameliorer_contraste(image_gris)
        images_conservees.append(Image.fromarray(image_amelioree))

    # Sauvegarde
    if images_conservees:
        images_conservees[0].save(
            chemin_tiff_sortie,
            save_all=True,
            append_images=images_conservees[1:],
            compression="tiff_deflate"
        )
        print(f"{len(images_conservees)} tranches valides sauvegardées dans : {chemin_tiff_sortie}")
    else:
        print("Aucune tranche valide détectée.")
    return chemin_tiff_sortie

# === Exécution ===


def compute_thickness_array(mask):
    dt = distance_transform_edt(mask)
    local_thickness = dt * 2
    return np.mean(local_thickness[local_thickness > 0])

def compute_distance(mask):
    dt = distance_transform_edt(1-mask)
    local_distance = dt * 2
    return np.mean(local_distance[local_distance > 0])

def bone_fraction (img):  #calcul fraction osseuse
    pix_n_sum = np.sum(img==1)   #nbre de pixels noirs
    pix_b_sum = np.sum(img==2)   #nbre de pixels blancs
    frac = pix_b_sum/(pix_n_sum + pix_b_sum)   #fraction PAR IMAGE  
    return frac

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

chemin_entree = '/home/ttantot/Documents/P2I/projet/E17_10um_1200_1200_1200.tif'
chemin_sortie = '/home/ttantot/Documents/P2I/projet/test_code.tif'
traiter_images_tiff(chemin_entree, chemin_sortie)

tif_path = '/home/ttantot/Documents/P2I/projet/test_code.tif'
img1=Image.open(tif_path)      #pourquoi pas image.open

Frac_tot = 0
cpt=0
for i in range (img1.n_frames):
    img1.seek(i)
    img = np.array(img1)
    
    im = detour_clean(img)
    
    Frac_tot += bone_fraction(im)
    cpt+=1
print(f'fraction volumique ={Frac_tot/cpt}')