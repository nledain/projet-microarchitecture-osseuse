import numpy as np
from skimage.filters import laplace, threshold_otsu
from skimage.exposure import equalize_adapthist
from tifffile import TiffFile, imwrite
from PIL import Image

def compute_otsu_thresholds(image_stack):
    """
    Compute automatic thresholds for blurriness and brightness using Otsu's method,
    using raw (non-normalized) images.
    """
    lap_var_list = []
    brightness_list = []

    for img in image_stack:
        img = img.astype(np.float32)  # Use raw pixel values (0–255 or 0–65535)

        lap_var = laplace(img).var()
        mean_intensity = img.mean()

        lap_var_list.append(lap_var)
        brightness_list.append(mean_intensity)

    lap_var_array = np.array(lap_var_list)
    brightness_array = np.array(brightness_list)

    blur_threshold = threshold_otsu(lap_var_array)
    brightness_threshold = threshold_otsu(brightness_array)

    return blur_threshold, brightness_threshold

def is_blurry(image, bluriness_threshold):
    """Check if the image is blurry using Laplacian variance."""
    return laplace(image).var() < bluriness_threshold

def is_too_dark(image, brightness_threshold):
    """Check if the image is too dark based on mean pixel value."""
    return np.mean(image) < brightness_threshold

def enhance_contrast(image):
    """Enhance contrast using adaptive histogram equalization (CLAHE)."""
    # Must convert to 8-bit and normalize to [0, 1] for CLAHE to work correctly
    img_norm = (image - image.min()) / (image.ptp() + 1e-8)
    return equalize_adapthist(img_norm, clip_limit=0.03)

def process_tiff_stack(chemin_tiff_entree, chemin_tiff_sortie, bluriness_threshold, brightness_threshold):
    image_tiff = Image.open(chemin_tiff_entree)
    images_conservees = []
    total_tranches = image_tiff.n_frames

    for indice in range(total_tranches):
        image_tiff.seek(indice)
        image_gris = np.array(image_tiff.convert('L'))

        # On teste l'image brute AVANT traitement
        if is_blurry(image_gris, bluriness_threshold) or is_too_dark(image_gris, brightness_threshold):
            continue  # on saute cette image

        # Si ok → on améliore l’image pour l’export
        image_amelioree = enhance_contrast(image_gris)
        images_conservees.append(Image.fromarray(image_amelioree))

    # Sauvegarde
    if images_conservees:
        images_conservees[0].save(
            chemin_tiff_sortie,
            save_all=True,
            append_images=images_conservees[1:]
        )
        print(f"{len(images_conservees)} tranches valides sauvegardées dans : {chemin_tiff_sortie}")
    else:
        print("Aucune tranche valide détectée.")
    return chemin_tiff_sortie

if __name__ == "__main__":
    # Input and output paths
    input_path = r"C:\Users\20109\Desktop\Echantillons\E10_10um_1200_1200_1200.tif"
    output_path = r"C:\Users\20109\Desktop\Echantillons\E10(filtered).tif"

    # Load stack to compute thresholds
    with TiffFile(input_path) as tif:
        stack = tif.asarray()

    # Compute Otsu thresholds using raw data
    blur_thresh, bright_thresh = compute_otsu_thresholds(stack)
    print(f"Otsu blur threshold: {blur_thresh:.4f}")
    print(f"Otsu brightness threshold: {bright_thresh:.4f}")

    # Process and save filtered stack
    process_tiff_stack(input_path, output_path, blur_thresh, bright_thresh)
