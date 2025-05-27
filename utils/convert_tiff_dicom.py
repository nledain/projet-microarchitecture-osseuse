import os
import numpy as np
from PIL import Image
import pydicom
from pydicom.uid import generate_uid
from pydicom.dataset import Dataset, FileDataset
import datetime

def convert_tiff_to_dicom(tiff_path, output_dir, pixel_spacing=(0.1, 0.1), slice_thickness=0.1):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the TIFF
    img = Image.open(tiff_path)
    n_frames = img.n_frames

    # Metadata
    series_instance_uid = generate_uid()
    study_instance_uid = generate_uid()
    sop_class_uid = pydicom.uid.SecondaryCaptureImageStorage

    for i in range(n_frames):
        img.seek(i)
        slice_array = np.array(img)

        file_meta = pydicom.Dataset()
        file_meta.MediaStorageSOPClassUID = sop_class_uid
        file_meta.MediaStorageSOPInstanceUID = generate_uid()
        file_meta.ImplementationClassUID = generate_uid()

        # Create dataset
        ds = FileDataset(f"slice_{i}.dcm", {}, file_meta=file_meta, preamble=b"\0" * 128)
        ds.Modality = 'OT'  # Other
        ds.ContentDate = str(datetime.date.today()).replace('-', '')
        ds.ContentTime = str(datetime.datetime.now().time()).replace(':', '').split('.')[0]

        ds.StudyInstanceUID = study_instance_uid
        ds.SeriesInstanceUID = series_instance_uid
        ds.SOPInstanceUID = file_meta.MediaStorageSOPInstanceUID
        ds.SOPClassUID = sop_class_uid

        ds.Rows, ds.Columns = slice_array.shape
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsStored = 8
        ds.BitsAllocated = 8
        ds.HighBit = 7
        ds.PixelRepresentation = 0
        ds.PixelSpacing = [str(pixel_spacing[0]), str(pixel_spacing[1])]
        ds.SliceThickness = str(slice_thickness)
        ds.InstanceNumber = i + 1
        ds.ImagePositionPatient = [0, 0, i * slice_thickness]

        ds.PixelData = slice_array.tobytes()

        ds.save_as(os.path.join(output_dir, f"slice_{i:04d}.dcm"))

    print(f"✅ DICOM series saved to: {output_dir}")

# === Example Usage ===
if __name__ == "__main__":
    tiff_path = "path/to/your/filtered_volume.tif"
    output_dir = "path/to/output/dicom_series"
    convert_tiff_to_dicom(tiff_path, output_dir)
