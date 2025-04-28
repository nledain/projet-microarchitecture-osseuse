# Prédiction du Risque de Fracture Osseuse

Ce projet est réalisé dans le cadre du cours P2I8 à l'INSA Lyon (2024-2025).

L'objectif est de prédire le risque de fracture d'échantillons osseux à partir d'images de tomographie aux rayons X, combinées à une modélisation par éléments finis (MEF).

## Objectifs du Projet

- Traiter les scans 3D de tomographie des échantillons osseux
- Calculer des paramètres architecturaux (BV/TV, épaisseur trabéculaire, etc.)
- Générer des maillages en éléments finis basés sur les voxels
- Associer des propriétés mécaniques à chaque voxel (densité → module de Young)
- Effectuer des simulations mécaniques pour prédire le risque de fracture

## Organisation du Projet

- `data/` — Données brutes et données traitées
- `notebooks/` — Notebooks Jupyter pour l'exploration et l'analyse
- `code/` — Scripts Python pour le traitement, la segmentation, le maillage, et l'export
- `requirements.txt` — Liste des bibliothèques Python nécessaires

## Auteurs

- Équipe G283
- INSA Lyon — P2I8 : Imagerie Industrielle et Médicale
