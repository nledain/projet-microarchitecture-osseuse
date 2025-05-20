import SimpleITK as sitk
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops, label
from scipy.linalg import eigvalsh

def read_image(path):
    image = sitk.ReadImage(path)
    return sitk.GetArrayFromImage(image)

def compute_BVTV(mask):
    return np.sum(mask) / np.prod(mask.shape)

def compute_thickness(mask):
    dt = distance_transform_edt(mask)
    local_thickness = dt * 2
    return np.mean(local_thickness[local_thickness > 0])

def compute_separation(inverted_mask):
    dt = distance_transform_edt(inverted_mask)
    local_sep = dt * 2
    return np.mean(local_sep[local_sep > 0])

def compute_anisotropy(mask):
    labeled = label(mask)
    props = regionprops(labeled)
    coords = np.vstack([p.coords for p in props if p.area > 10])
    coords = coords - coords.mean(axis=0)
    cov = np.cov(coords.T)
    eigvals = np.sort(eigvalsh(cov))[::-1]
    return eigvals[0] / eigvals[-1] if eigvals[-1] > 0 else np.nan

def compute_porosity(mask, original_bone_mask):
    return 1 - (np.sum(mask) / np.sum(original_bone_mask))

def analyze_region(mask, original, region_name="trabecular"):
    results = {}

    BVTV = compute_BVTV(mask)
    thickness = compute_thickness(mask)
    anisotropy = compute_anisotropy(mask)

    results[f"{region_name}_BVTV"] = BVTV
    results[f"{region_name}_Thickness"] = thickness
    results[f"{region_name}_Anisotropy"] = anisotropy

    if region_name == "trabecular":
        separation = compute_separation(~mask)
        results[f"{region_name}_Separation"] = separation
    else:
        porosity = compute_porosity(mask, original)
        results[f"{region_name}_Porosity"] = porosity

    return results

def main():
    # === Load images ===
    bone_image = read_image("bone_scan.tiff")
    cortical_mask = read_image("cortical_mask.tiff").astype(bool)
    trabecular_mask = read_image("trabecular_mask.tiff").astype(bool)

    # === Analyze ===
    trabecular_results = analyze_region(trabecular_mask, bone_image, "trabecular")
    cortical_results = analyze_region(cortical_mask, bone_image, "cortical")

    # === Save results ===
    all_results = {**trabecular_results, **cortical_results}
    df = pd.DataFrame([all_results])
    df.to_excel("bone_analysis_results.xlsx", index=False)
    print("✅ Results saved to 'bone_analysis_results.xlsx'")

if __name__ == "__main__":
    main()
