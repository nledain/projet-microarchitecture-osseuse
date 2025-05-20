import cv2
import numpy as np
from PIL import Image

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
chemin_entree = "D:/INSA/FIMI 3-a/P2I/E17_10um_1200_1200_1200.tif"
chemin_sortie = "D:/INSA/FIMI 3-a/P2I/E17_FILTRÉ.tif"
traiter_images_tiff(chemin_entree, chemin_sortie)
