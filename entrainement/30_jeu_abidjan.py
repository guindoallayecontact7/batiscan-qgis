"""Vignettes d'entraînement à partir de l'orthophoto drone d'Abidjan et des bâtiments dessinés à la main.

python-qgis 30_jeu_abidjan.py <orthophoto.tif> <batiments.gpkg> <carres.gpkg> <dossier_jeu> [<non_tranches.gpkg>]

Une vignette de 512 px à 0,3 m (153,6 m) est centrée sur chaque carré de 100 m vérifié.
Masque : 1 intérieur, 2 contour des bâtiments dessinés ; 0 le reste du carré ;
255 hors du carré (non vérifié), hors de l'orthophoto et sur les détections v6 non tranchées
(objets laissés en rouge par l'utilisateur, ni dessinés ni refusés). Dossiers <carré>_drone.
Les géométries invalides sont réparées (makeValid) ; les fichiers d'entrée ne sont pas modifiés.
"""

import os
import sys

from commun_qgis import demarrer_qgis

app = demarrer_qgis()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from qgis.core import QgsRasterLayer, QgsRectangle, QgsVectorLayer  # noqa: E402

from batiscan.core.image_source import SourceImage  # noqa: E402

ortho, fbat, fcar, sortie = sys.argv[1:5]
ortho = os.environ.get("ORTHO", ortho)  # sous Windows, le lanceur .bat de QGIS casse les chemins accentués
T, RES = 512, 0.3
lay = QgsRasterLayer(ortho, "ortho")
src = SourceImage(lay)
bat = [f.geometry().makeValid() for f in QgsVectorLayer(fbat, "b", "ogr").getFeatures()]
car = QgsVectorLayer(fcar, "c", "ogr")
nt = [f.geometry().makeValid() for f in QgsVectorLayer(sys.argv[5], "n", "ogr").getFeatures()] if len(sys.argv) > 5 else []
noyau = np.ones((3, 3), np.uint8)


def remplir(m, g, x0, y0, val):
    for poly in (g.asMultiPolygon() if g.isMultipart() else [g.asPolygon()]):
        if poly:
            a = [np.array([[(q.x() - x0) / RES, (y0 - q.y()) / RES] for q in r], np.int32) for r in poly]
            cv2.fillPoly(m, [a[0]], val)
            for trou in a[1:]:
                cv2.fillPoly(m, [trou], 0)


for fc in car.getFeatures():
    nom, cadre = fc["nom"], fc.geometry()
    c = cadre.centroid().asPoint()
    h = T * RES / 2
    x0, y0 = c.x() - h, c.y() + h
    img = src.read(QgsRectangle(x0, y0 - 2 * h, x0 + 2 * h, y0), T, T)
    alpha = img[..., :3].sum(axis=2) > 0
    inst = np.zeros((T, T), np.int32)
    k = 0
    for g in bat:
        if g.intersects(cadre):
            k += 1
            remplir(inst, g, x0, y0, k)
    dedans = np.zeros((T, T), np.uint8)
    remplir(dedans, cadre, x0, y0, 1)
    b = inst > 0
    lab = inst.astype(np.float32)
    bord = cv2.dilate(lab, noyau) != cv2.erode(lab, noyau)
    bord = cv2.dilate(bord.astype(np.uint8), noyau).astype(bool) & cv2.dilate(b.astype(np.uint8), noyau).astype(bool)
    m = np.zeros((T, T), np.uint8)
    m[b] = 1
    m[bord] = 2
    m[(dedans == 0) & ~b] = 255  # bâtiment débordant du carré : gardé en entier
    ign = np.zeros((T, T), np.uint8)
    for g in nt:
        if g.intersects(cadre):
            remplir(ign, g, x0, y0, 1)
    m[(cv2.dilate(ign, noyau, iterations=2) > 0) & ~b] = 255
    m[~alpha] = 255
    d = os.path.join(sortie, f"{nom}_drone")
    os.makedirs(d, exist_ok=False)
    cv2.imwrite(os.path.join(d, "img_000_000.png"), img[..., 2::-1] if img.shape[2] >= 3 else img)
    cv2.imwrite(os.path.join(d, "msk_000_000.png"), m)
    print(f"{nom} ({fc['role']}) : {k} bâtiments, {int(b.sum() * RES * RES)} m² de bâti, "
          f"{int(((m == 0)).sum() * RES * RES)} m² de fond", flush=True)
