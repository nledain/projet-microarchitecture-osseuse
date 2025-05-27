#Parcours des fichiers .tif 
from PIL import Image
import matplotlib.pyplot as plt

# Ouvre le fichier TIFF multi-pages
tiff_path = "D:\Documents-Thibaud\Mes-Docs\insa\COURS 2A\P2I8\Projet\E17_10um_1200_1200_1200.tif"
img = Image.open(tiff_path)

# Parcourir chaque image
# for i in range(img.n_frames):
#     img.seek(i)  # Aller à la i-ème image
#     plt.imshow(img, cmap="gray")
#     plt.title(f"Page {i + 1}")
#     plt.axis("off")
    
img.seek(806)
plt.imshow(img,cmap='gray')