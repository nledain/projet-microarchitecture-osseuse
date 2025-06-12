import numpy as np
import tifffile
from tqdm import tqdm
from utils.trinarisation2 import detour_clean_stack
from utils.binarisation3 import separate_trabecular_cortical_by_thickness
from utils.propreties4 import trabecular_thickness, trabecular_distance, bone_fraction, anisotropy

def run_full_analysis(grayscale_tif_path):
    print(f"Reading grayscale TIF: {grayscale_tif_path}")
    raw_volume = tifffile.imread(grayscale_tif_path)

    print("Step 1: Trinarizing slices...")
    trinarized = detour_clean_stack(raw_volume)

    print("Step 2: Separating trabecular and cortical bone...")
    labeled_volume = separate_trabecular_cortical_by_thickness(trinarized, threshold_px=80)

    print("Step 3: Computing metrics...")

    bvtv = bone_fraction(trinarized)
    tbth = trabecular_thickness(labeled_volume)
    tbsp = trabecular_distance(labeled_volume)

    print("Step 4: Computing anisotropy...")
    anisotropy(grayscale_tif_path)

    print("\n Results Summary:")
    print(f"BV/TV (Bone Volume Fraction): {bvtv:.4f}")
    print(f"Tb.Th (Trabecular Thickness): {tbth:.2f} px")
    print(f"Tb.Sp (Trabecular Spacing):   {tbsp:.2f} px")

if __name__ == "__main__":
    input_path = r"C:\Users\20109\Desktop\Echantillons\E10_10um_1200_1200_1200.tif"
    run_full_analysis(input_path)
