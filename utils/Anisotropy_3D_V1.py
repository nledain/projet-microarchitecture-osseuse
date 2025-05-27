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

## Now the computation of the Fractional Anisotropy
 
# Extract eigenvalues

lambda1 = val[..., 0] ## we take the first eigenvalue (the largest) from the 1st dimension of the val array, for all voxels
lambda2 = val[..., 1] 
lambda3 = val[..., 2] ## smallest eigenvalue for each voxel

## formula of the FA 
lambda_mean = (lambda1 + lambda2 + lambda3) / 3

numerator = ((lambda1 - lambda_mean)**2 +
             (lambda2 - lambda_mean)**2 +
             (lambda3 - lambda_mean)**2)

denominator = lambda1**2 + lambda2**2 + lambda3**2 + 1e-12

FA = np.sqrt(1.5) * np.sqrt(numerator) / np.sqrt(denominator)  # Shape (Z, Y, X)

# Print summary stats, le .4f indique que l'on veut un float avec 4 chiffres après la virgule
print(f"Fractional Anisotropy: mean = {FA.mean():.4f}, median = {np.median(FA):.4f}, max = {FA.max():.4f}")

## we plot the histogram of Fractional Anisotropy
plt.hist(FA.ravel(), bins=100, color='steelblue')
plt.title("Histogram of Fractional Anisotropy (FA)")
plt.xlabel("FA value")
plt.ylabel("Voxel count")
plt.grid(True)
plt.tight_layout()
plt.show()




