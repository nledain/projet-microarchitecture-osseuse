import matplotlib.pyplot as plt
from structure_tensor import eig_special_3d, structure_tensor_3d
from scipy.ndimage import distance_transform_edt
from skimage.transform import downscale_local_mean
import numpy as np
import tifffile



def trabecular_thickness(volume, start=0, end=None):
    volume = volume[start:end]
    volume = np.where(volume == 2, 1, 0)
    dt = distance_transform_edt(volume)
    local_thickness = dt * 2
    return np.mean(local_thickness[local_thickness > 0])


def trabecular_distance(volume, start=0, end=None):
    volume = volume[start:end]
    volume = np.where(volume == 1, 1, 0)
    dt = distance_transform_edt(volume)
    local_distance = dt * 2
    return np.mean(local_distance[local_distance > 0])


def bone_fraction (img_arr):  #calcul fraction osseuse
    pix_n_sum = np.sum(img_arr==1)   #nbre de pixels noirs
    pix_b_sum = np.sum(img_arr==2)   #nbre de pixels blancs
    frac = pix_b_sum/(pix_n_sum + pix_b_sum)   #fraction PAR IMAGE  
    return frac



def frac_vol_tot (volume)
    Frac = 0
    cpt = 0
    im_trin = Image.fromarray(volume)
    for i in range (im_trin.n_frames):
        im_trin.seek(i)
        im_trin_arr = np.array(im_trin)
        
        Frac += bone_fraction(im_trin_arr)
        cpt+=1
    Frac_tot=frac/cpt
    print(f'fraction volumique ={Frac_tot}')






def bone_fraction_3D(volume_3d, start=0, end=None):
    """
    Computes bone volume fraction (BV/TV) over a 3D trinarized volume.

    Parameters:
        volume_3d (np.ndarray): 3D array with values:
            0 = background,
            1 = non-bone (inside),
            2 = bone
        start (int): first slice to include (default: 0)
        end (int): last slice to include (default: full volume)

    Returns:
        float: bone fraction BV/TV = volume of bone / (bone + marrow)
    """
    subvolume = volume_3d[start:end]  # shape (Z, H, W)

    num_empty_voxels = np.sum(subvolume == 1)
    num_bone_voxels = np.sum(subvolume == 2)

    total = num_empty_voxels + num_bone_voxels
    if total == 0:
        return 0.0  # avoid division by zero

    return num_bone_voxels / total



def anisotropy(tiff_file, start=0, end=None):
    data_1 = tifffile.imread(tiff_file).astype(np.float32)

    # Apply slice range
    data_1 = data_1[start:end]

    # Downscale
    downscale_factors = (2, 2, 2)
    z, y, x = data_1.shape
    z2, y2, x2 = [d - d % f for d, f in zip(data_1.shape, downscale_factors)]
    data_cropped = data_1[:z2, :y2, :x2]
    data_downscaled = downscale_local_mean(data_cropped, downscale_factors).astype(np.float32)

    print(f"Forme de l'échantillon: {data_1.shape}") 

    sigma = 1
    rho = 2

    S = structure_tensor_3d(data_downscaled, sigma, rho)
    val, vec = eig_special_3d(S)

    lambda1 = val[..., 0]
    lambda2 = val[..., 1]
    lambda3 = val[..., 2]

    lambda_mean = (lambda1 + lambda2 + lambda3) / 3

    numerator = ((lambda1 - lambda_mean)**2 +
                 (lambda2 - lambda_mean)**2 +
                 (lambda3 - lambda_mean)**2)

    denominator = lambda1**2 + lambda2**2 + lambda3**2 + 1e-12

    FA = np.sqrt(1.5) * np.sqrt(numerator) / np.sqrt(denominator)

    print(f"Fraction anisotropique: moyenne = {FA.mean():.4f}, mediane = {np.median(FA):.4f}, maximum = {FA.max():.4f}")

    plt.hist(FA.ravel(), bins=100, color='steelblue')
    plt.title("Histogramme de la fraction anisotropique (FA)")
    plt.xlabel("valeur de la FA")
    plt.ylabel("Nombre de Voxels")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
