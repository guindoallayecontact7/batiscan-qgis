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
| Orthophoto drone : vignettes à partir de bâtiments dessinés à la main dans des carrés de 100 m | `30_jeu_abidjan.py` | Python de QGIS |
| Orthophoto drone : score par objet (IoU 0,5 et couverture 30 %) sur les carrés de test | `31_score_abidjan.py` | Python de QGIS |
| Recalage de masques décalés sur les prédictions d'un modèle | `06_recaler.py` | Python avec PyTorch |
| Mesure par village, au pixel et par bâtiment | `07_mesurer.py` | Python avec PyTorch |
| Repérage des fausses détections sur un jeu | `12_negatifs.py` | Python avec PyTorch |

`00_open_buildings.py` extrait aussi les emprises Google/Microsoft Open Buildings autour des villages, pour comparaison.

## Masques

Le masque a trois classes : fond (0), intérieur de bâtiment (1) et contour (2). La classe contour sépare les bâtiments qui se touchent : à la détection, les intérieurs servent de graines et chaque bâtiment reprend son contour.

La valeur 255 marque les zones ignorées pendant l'entraînement :

- les abords des points restés sans emprise ;
- les emprises de fiabilité « faible » ;
- les emprises surtout couvertes de végétation (SAM a pris un arbre) ;
- les grandes formes libres de plus de 250 m².

Seules les vignettes situées à moins de 60 m d'un point sont gardées, pour ne pas apprendre comme « fond » des bâtiments hors de la zone relevée.

## Reprendre un entraînement

`03_entrainer.py` accepte `--init` (poids de départ), `--lr`, `--jeux-sup` (vignettes en plus) et `--sup-seul` (pré-entraînement sur ces seules vignettes). La valeur 3 d'un masque marque du fond que le modèle prend pour du bâti ; `--poids-dur` règle son poids. Le fichier `dernier.pt` garde la fin de l'entraînement, à côté de `meilleur.pt`.

Avant d'apprendre des fausses détections comme du fond, regardez-les : une référence incomplète en contient beaucoup qui sont de vrais toits.

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
