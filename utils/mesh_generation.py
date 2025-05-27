import SimpleITK as sitk
import numpy as np
from skimage import measure
from skimage.transform import downscale_local_mean
import vtk
from vtk.util import numpy_support
from tqdm import tqdm
import time

start_total = time.time()

# Step 1: Load the 3D TIFF image
print("Loading TIFF image...")
start = time.time()
image = sitk.ReadImage(traiter_images_tiff("C:/Users/20109/Desktop/P2I8 Data/E10_10um_1200_1200_1200.tif", "C:/Users/20109/Desktop/P2I8 Data/E17_FILTRÉ.tif") , sitk.sitkFloat32)
arr = sitk.GetArrayFromImage(image)
print(f"Image loaded in {time.time() - start:.2f} seconds")

# Step 2: Downsample the volume to reduce computation (factor 10x in each dimension)
print("Downsampling image for performance...")
start = time.time()
arr_down = downscale_local_mean(arr, (10, 10, 10)).astype(np.float32)
print(f"Downsampled shape: {arr_down.shape} in {time.time() - start:.2f} seconds")

# Step 3: Threshold to segment bone
print("Applying threshold to extract bone...")
start = time.time()
binary = (arr_down > 150).astype(np.uint8)  # Adjust threshold as needed
print(f"Thresholding done in {time.time() - start:.2f} seconds")

# Step 4: Generate mesh from volume
print("Running marching cubes...")
start = time.time()
verts, faces, _, _ = measure.marching_cubes(binary, level=0)
print(f"Mesh generated with {len(verts)} vertices and {len(faces)} faces in {time.time() - start:.2f} seconds")

# Step 5: Convert to VTK PolyData
print("Converting mesh to VTK format...")
start = time.time()
points = vtk.vtkPoints()
points.SetData(numpy_support.numpy_to_vtk(verts.astype(np.float32)))

triangles = vtk.vtkCellArray()
for face in tqdm(faces, desc="Creating VTK triangles", unit="tri"):
    triangle = vtk.vtkTriangle()
    for i in range(3):
        triangle.GetPointIds().SetId(i, face[i])
    triangles.InsertNextCell(triangle)

polydata = vtk.vtkPolyData()
polydata.SetPoints(points)
polydata.SetPolys(triangles)
print(f"VTK mesh created in {time.time() - start:.2f} seconds")

# Step 6: Write binary STL
print("Saving to binary STL file...")
start = time.time()
writer = vtk.vtkSTLWriter()
writer.SetFileName("C:/Users/20109/Desktop/bone_model(3).stl")
writer.SetFileTypeToBinary()  # Faster and smaller file
writer.SetInputData(polydata)
writer.Write()
print(f"Binary STL saved in {time.time() - start:.2f} seconds")

# Final time
print(f"Total process completed in {time.time() - start_total:.2f} seconds")
