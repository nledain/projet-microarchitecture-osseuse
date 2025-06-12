import matplotlib.pyplot as plt
from structure_tensor import eig_special_3d, structure_tensor_3d
from scipy.ndimage import distance_transform_edt
from skimage.transform import downscale_local_mean
import numpy as np
import tifffile


def trabecular_thickness(volume):
    # Set everything except 2 to 0
    volume = np.where(volume == 2, 1, 0)
    dt = distance_transform_edt(volume)
    local_thickness = dt * 2
    return np.mean(local_thickness[local_thickness > 0])

def trabecular_distance(volume):
    # Set everything except 1 to 0
    volume = np.where(volume == 1, 1, 0)
    dt = distance_transform_edt(volume)
    local_distance = dt * 2
    return np.mean(local_distance[local_distance > 0])

def bone_fraction(image):  #calcul fraction osseuse
    pix_n_sum = np.sum(image==1)   #nbre de pixels noirs
    pix_b_sum = np.sum(image==2)   #nbre de pixels blancs
    bone_fraction = pix_b_sum/(pix_n_sum + pix_b_sum)   #fraction PAR IMAGE  
    return bone_fraction

def anisotropy(tiff_file):
    '''
    Parameters
    ----------
    tiff_file : string
        This is the path leading to the tiif file you want to analyze

    Prints
    -------
    - The shape of your tiff file (height, width, lenght)
    - Mean value of the Fractional Anistropy for the whole sample 
    - Median of all the Fractional Anistropies calculated
    - Maximum Fractional Anistropy 
    - Histogram of the Voxel Count in function of the Fractional Anisotropy

    '''
    ## on lit le fichier tiff en float32 directement pour réduire sa taille  
    ## cependant les travées très fines peuvent disparaître
    data_1 = tifffile.imread(tiff_file).astype(np.float32) 

    # Réduire par un facteur de 2 sur les 3 axes (X, Y, Z)
    downscale_factors = (2, 2, 2)

    # Si les dimensions ne sont pas divisibles par 2, on fait en sorte qu'elle puisse l'être
    z, y, x = data_1.shape
    z2, y2, x2 = [d - d % f for d, f in zip(data_1.shape, downscale_factors)]
    data_cropped = data_1[:z2, :y2, :x2]

    #Reduire en utilisant une moyenne locale 
    data_downscaled = downscale_local_mean(data_cropped, downscale_factors).astype(np.float32)
    
    ## dimensions d el'image 3D dans toutes les directions 
    print(f"Forme de l'échantillon: {data_1.shape}") 

    ## sigma et rho déterminent la sensibilité à l’échelle spatiale dans le calcul du tenseur de structure 

    sigma = 1 ## est appliqué avant le calcul du tenseur, il filtre le bruit et calcule des gradients
              ## plus il est grand plus on va lisser l'image et donc omettre les petits détails 
    rho = 2 ## moyenne les variations du tenseur dans une région
            ## plus rho est grand plus le tenseur est influencé par ses voisins et donc plus on consomme de la puissnace de calcul

    ## pour limiter le temps d'interprétation de l'image 3D, il faut donc un grand sigma et un petit rho 

    S = structure_tensor_3d(data_downscaled, sigma, rho)
    val, vec = eig_special_3d(S)

    ## Maintenant on détermine les différentes valeurs voulues sur l'Anisotropie
 
    # Extraire les valeurs propres du structure tensor pour chaque Voxel

    lambda1 = val[..., 0] ## on prend la plus grand eigenvalue qui détermine la direction favorisée, une dimension en particulier
    lambda2 = val[..., 1] 
    lambda3 = val[..., 2] ## on prend la plus petite valeur propre

    ## Formule de la fraction anisotropique  
    lambda_mean = (lambda1 + lambda2 + lambda3) / 3

    numerator = ((lambda1 - lambda_mean)**2 +
             (lambda2 - lambda_mean)**2 +
             (lambda3 - lambda_mean)**2)

    denominator = lambda1**2 + lambda2**2 + lambda3**2 + 1e-12

    FA = np.sqrt(1.5) * np.sqrt(numerator) / np.sqrt(denominator)  # Shape (Z, Y, X)

    # On print , le .4f indique que l'on veut un float avec 4 chiffres après la virgule
    print(f"Fraction anisotropique: moyenne = {FA.mean():.4f}, mediane = {np.median(FA):.4f}, maximum = {FA.max():.4f}")

    ## we plot the histogram of Fractional Anisotropy
    plt.hist(FA.ravel(), bins=100, color='steelblue')
    plt.title("Histograme de la fraction anistropique (FA)")
    plt.xlabel("valeur de la FA")
    plt.ylabel("Nombre de Voxels")
    plt.grid(True)
    plt.tight_layout()
    plt.show()