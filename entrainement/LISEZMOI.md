# Entraîner un modèle de détection des bâtiments

Ces scripts transforment des points de bâtiments relevés sur le terrain en un modèle qui détecte les bâtiments sur l'image satellite, sans points. Le modèle obtenu se charge dans l'onglet « Automatique » de BatiScan.

## Chaîne de traitement

| Étape | Script | Python utilisé |
|---|---|---|
| 1. Emprises SAM à partir des points | `01_emprises_sam.py` (ou `lancer_emprises.sh` pour plusieurs villages et fonds) | Python de QGIS |
| 1 bis. Emprises à partir de polygones existants (TerraLab...) | `01b_polygones.py` | Python de QGIS |
| 2. Vignettes image + masque | `02_jeu.py` (ou `lancer_jeu.sh`) | Python de QGIS |
| 3. Entraînement U-Net | `03_entrainer.py` | Python avec PyTorch et GPU |
| 4. Export ONNX avec métadonnées Deepness | `04_exporter_onnx.py` | Python avec PyTorch |
| 5. Détection sur un village test et comparaison aux points | `05_evaluer.py` | Python de QGIS |

`00_open_buildings.py` extrait aussi les emprises Google/Microsoft Open Buildings autour des villages, pour comparaison.

## Masques

Le masque a trois classes : fond (0), intérieur de bâtiment (1) et contour (2). La classe contour sépare les bâtiments qui se touchent : à la détection, les intérieurs servent de graines et chaque bâtiment reprend son contour.

La valeur 255 marque les zones ignorées pendant l'entraînement :

- les abords des points restés sans emprise ;
- les emprises de fiabilité « faible » ;
- les emprises surtout couvertes de végétation (SAM a pris un arbre) ;
- les grandes formes libres de plus de 250 m².

Seules les vignettes situées à moins de 60 m d'un point sont gardées, pour ne pas apprendre comme « fond » des bâtiments hors de la zone relevée.

## Environnement PyTorch sous Windows

L'entraînement demande PyTorch avec CUDA. Avec une carte RTX 50xx, il faut la version CUDA 12.8 ou plus récente :

```bash
uv pip install --target libs torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install --target libs segmentation-models-pytorch onnx onnxruntime opencv-python-headless
```

Si Windows bloque l'exécutable d'un environnement virtuel (Smart App Control), installer dans un dossier `libs` comme ci-dessus et lancer l'interpréteur de base avec `PYTHONPATH=libs`.

## Exemple

```bash
bash lancer_emprises.sh "<dossier des points>" travail/emprises
bash lancer_jeu.sh travail/emprises travail/jeu
python 03_entrainer.py travail/jeu travail/modele --validation Doumanani --epoques 80
python 04_exporter_onnx.py travail/modele/meilleur.pt travail/batiscan_sahel.onnx
python-qgis 05_evaluer.py travail/batiscan_sahel.onnx "<points Doumanani.shp>" google travail/test.gpkg
```

Un village entier est réservé à la validation : il mesure ce que le modèle fait sur une zone qu'il n'a jamais vue.

## Imagerie

Les conditions d'utilisation de Google et de Bing limitent l'extraction de données à partir de leurs images. Pour un usage commercial, entraîner et détecter sur une imagerie dont la licence le permet.
