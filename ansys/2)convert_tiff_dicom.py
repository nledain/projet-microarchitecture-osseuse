import SimpleITK as sitk
import numpy as np
import os

def tiff_to_dicom(tiff_path, output_dir):
    # Load 3D TIFF
    image = sitk.ReadImage(tiff_path)
    array = sitk.GetArrayFromImage(image)  # shape: [slices, height, width]

    os.makedirs(output_dir, exist_ok=True)

    for i, slice_ in enumerate(array):
        slice_image = sitk.GetImageFromArray(slice_[np.newaxis, :, :])
        slice_image.SetSpacing(image.GetSpacing())
        slice_image.SetOrigin(image.GetOrigin())
        slice_image.SetDirection(image.GetDirection())

        dicom_path = os.path.join(output_dir, f"slice_{i:03}.dcm")
        sitk.WriteImage(slice_image, dicom_path)

    print(f"Converted {len(array)} slices to DICOM in {output_dir}")

if __name__ == "__main__":
    input_path = "path/to/your/tiff_file.tiff"
    output_path = "path/to/output/dicom_directory"
    tiff_to_dicom(input_path, output_path)
