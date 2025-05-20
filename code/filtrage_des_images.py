from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2  # OpenCV pour le filtre de flou

# === PARAMÈTRES ===
tiff_path = "D:\\INSA\\FIMI 3-a\\P2I\\E17_10um_1200_1200_1200.tif"
FLU_BLUR_THRESHOLD = 100.0   # seuil pour détecter le flou (variance du Laplacien)
INTENSITY_THRESHOLD = 15     # seuil minimum pour considérer l'image comme utile (0-255)

# === FONCTIONS UTILITAIRES ===

def is_blurry(np_image, threshold=FLU_BLUR_THRESHOLD):
    # Calcul de la variance du Laplacien pour détecter le flou
    laplacian_var = cv2.Laplacian(np_image, cv2.CV_64F).var()
    return laplacian_var < threshold

def is_too_dark(np_image, threshold=INTENSITY_THRESHOLD):
    # Pourcentage de pixels au-dessus d’un seuil d’intensité
    ratio_above_thresh = np.mean(np_image > threshold)
    return ratio_above_thresh < 0.05  # Moins de 5% de pixels utiles → image rejetée

def afficher_images_utiles(image, start=0, end=None):
    image.seek(0)  # retour au début
    end = end if end else image.n_frames

    images_valides = []
    for i in range(start, end):
        image.seek(i)
        img_np = np.array(image.convert('L'))  # niveau de gris

        if i < 200 or i > (image.n_frames - 200):  # tranches sujettes au flou
            if is_blurry(img_np) or is_too_dark(img_np):
                continue  # image rejetée
        # sinon, image valide
        images_valides.append((i, img_np))

    print(f"{len(images_valides)} images valides détectées parmi les {end - start} analysées.")

    # Affichage de quelques images valides (par exemple 5 réparties)
    n = len(images_valides)
    if n == 0:
        print("Aucune image exploitable.")
        return

    plt.figure(figsize=(15, 5))
    for idx, (i, img_np) in enumerate(images_valides[::max(1, n // 5)][:5]):
        plt.subplot(1, 5, idx + 1)
        plt.imshow(img_np, cmap='gray')
        plt.title(f"Slice {i}")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    return images_valides

# === EXECUTION ===
img = Image.open(tiff_path)
valid_images = afficher_images_utiles(img)
