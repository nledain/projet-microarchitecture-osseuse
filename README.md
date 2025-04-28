# Bone Fracture Risk Prediction

This project is part of the P2I8 course at INSA Lyon (2024-2025).

We aim to predict the fracture risk of bone samples using 3D X-ray tomography imaging combined with finite element modeling (FEM).

## Project Objectives

- Process 3D X-ray tomography scans of bone samples
- Calculate architectural parameters (BV/TV, trabecular thickness, etc.)
- Generate voxel-based finite element meshes
- Assign mechanical properties to each voxel (density → Young's modulus)
- Perform mechanical simulations to predict fracture risk

## Project Structure

- `data/` — Raw and processed image data
- `notebooks/` — Jupyter notebooks for exploration and analysis
- `code/` — Python scripts for processing, segmentation, meshing, and exporting
- `requirements.txt` — List of Python libraries required

## Authors

- Team G283
- INSA Lyon — P2I8: Imaging in Industry and Medicine
