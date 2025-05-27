import numpy as np
import qctma


dicom_dir_path = r"D:\MEKANOS auto soft_2024-02-12\tests\test_assets_femur\Dicoms"  # Chemin du dossier contenant les dicoms
source_mesh_path = r"D:\MEKANOS auto soft_2024-02-12\tests\test_assets_femur\Simulation_Segmentation_Femur_Lyon\Segmentation.cdb"  # Chemin du maillage source
deltaE = 50  # Pas de variation du module d'Young
out_mesh_path = r"D:\Marc\Windows folders\Desktop\Test_QCTMA\Segmentation.cdb"  # Chemin du maillage de sortie (doit finir par ".cdb")
smoothing = 0  # Paramètre de lissage gaussien (0 pour aucun lissage)
window_smoothing = 0  # Taille du kernel pour le lissage gaussien
nb_process = 10  # Nombre de processus à utiliser pour le traitement

def gl2density(gl_array):
    """
    Fonction de conversion des niveaux de gris en densité.
    :param gl_array: List des niveaux de gris
    :return: List des densités
    """
    rho = 0.0077*gl_array - 0.3726
    rho[rho <= 0] = 0.01
    return rho


def density2E(rho):
    """
    Fonction de conversion de la densité en module d'Young.
    :param rho: List des densités
    :return: List des modules d'Young
    """
    E_mat = 2100*rho*10**3-80
    E_mat = np.array(E_mat)
    E_mat[E_mat < 0.01] = 0.01
    return E_mat


if __name__ == "__main__":  # !!! Ne pas oublier de mettre le code dans cette condition pour éviter les erreurs dû au multiprocessing
    qctma.qctma(dcm_path=dicom_dir_path, mesh_path=source_mesh_path, gl2density=gl2density,
                density2E=density2E, deltaE=deltaE, process=True, save_mesh_path=out_mesh_path,
                gaussian_smoothing=smoothing, window_gaussian_smoothing=window_smoothing,
                nb_process=nb_process)
    
    

