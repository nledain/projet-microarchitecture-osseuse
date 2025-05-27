# -*- coding: utf-8 -*-
"""
Created on Tue May 27 08:54:05 2025

@author: Baptiste
"""

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
import tifffile

import plotly
import plotly.express as px
import skimage as ski

import cv2
import numpy as np
from PIL import Image

from skimage.feature import structure_tensor, structure_tensor_eigenvalues
from structure_tensor import eig_special_3d, structure_tensor_3d

from skimage.transform import downscale_local_mean
from skimage import img_as_float32

## en premier lieu on filtre le tiff pour se débarasser des premières images non importantes
## le code ci-dessous enregistre un nouveau fichier sous le nom de Test_1_filtré qui ne comporte que les images nécessaires
''' 
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
chemin_entree = "Test_1.tif"
chemin_sortie = "Test_1_filtré.tif"
traiter_images_tiff(chemin_entree, chemin_sortie)
'''

## on lit l'image en 3 dimensions

data_1 = tifffile.imread("Test_1_filtré.tif").astype(np.float32) ## on lit le fichier tiff en float32 directement cependant les travées très fines peuvent disparaître

# Downscale by factor 2 on all 3 axes (Z, Y, X)
downscale_factors = (2, 2, 2)

# If dimensions are not divisible by 2, crop to make them fit
z, y, x = data_1.shape
z2, y2, x2 = [d - d % f for d, f in zip(data_1.shape, downscale_factors)]
data_cropped = data_1[:z2, :y2, :x2]

# Downscale using local mean (preserves intensity structure)
data_downscaled = downscale_local_mean(data_cropped, downscale_factors).astype(np.float32)

print(f'shape: {data_1.shape}') ## -> dimensions of the 3D image in each direction

## sigma et rho déterminent la sensibilité à l’échelle spatiale dans le calcul du tenseur de structure 

sigma = 1 ## est appliqué avant le calcul du tenseur, il filtre le bruit et calcule des gradients / plus il est grand plus on va lisser l'image et donc omettre les petits détails 
rho = 2 ## moyenne les variations du tenseur dans une région, plus rho est grand plus le tenseur est influencé par ses voisins et donc plus on consomme de la puissnace de calcul

## pour limiter le temps d'interprétation de l'image 3D, il faut donc un grand sigma et un petit rho 

S = structure_tensor_3d(data_downscaled, sigma, rho)
val, vec = eig_special_3d(S)

print(f'The eigenvalues are {val}')




