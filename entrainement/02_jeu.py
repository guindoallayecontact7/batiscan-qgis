"""Étape 2 : vignettes image + masque à partir des emprises SAM d'un village.

python-qgis 02_jeu.py <emprises.gpkg> <google|bing> <dossier_sortie> [taille=512] [res=0.3]

Masque à 3 classes, plus une valeur ignorée :
  0 = fond, 1 = intérieur de bâtiment, 2 = contour de bâtiment (sépare les
  bâtiments qui se touchent), 255 = ignoré à l'entraînement.
Sont ignorés : les abords des points restés sans emprise (un bâtiment est là
mais n'a pas été détouré), les emprises de fiabilité « faible », celles qui
sont surtout couvertes de végétation (SAM a pris la couronne d'un arbre posé
sur le point) et les grandes taches irrégulières (cour, concession entière).
Seules les vignettes proches des points sont gardées : hors du village, un
bâtiment non relevé serait appris comme du fond.
"""

import os
import sys

from commun_qgis import couche_fond, demarrer_qgis

app = demarrer_qgis()

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from qgis.core import (  # noqa: E402
    QgsCoordinateTransform,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsRectangle,
    QgsSpatialIndex,
    QgsVectorLayer,
)

from batiscan.core.image_source import SourceImage  # noqa: E402

gpkg, fond, sortie = sys.argv[1:4]
T = int(sys.argv[4]) if len(sys.argv) > 4 else 512
RES = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
DIST_VILLAGE = 60.0  # m : une vignette est gardée si son centre est à moins de 60 m d'un point
RAYON_IGNORE = 6.0  # m autour d'un point sans emprise
PART_VEGETATION_MAX = 0.2
AIRE_LIBRE_MAX = 250.0  # m² : au-delà, une forme « libre » est douteuse

emp = QgsVectorLayer(f"{gpkg}|layername=emprises", "e", "ogr")
pts = QgsVectorLayer(f"{gpkg}|layername=points", "p", "ogr")
src = SourceImage(couche_fond(fond))
tr = QgsCoordinateTransform(emp.crs(), src.crs, QgsProject.instance())

polys, faibles = [], []
for f in emp.getFeatures():
    g = QgsGeometry(f.geometry())
    g.transform(tr)
    douteux = f["fiabilite"] == "faible" or (f["forme"] == "libre" and f["surface_m2"] > AIRE_LIBRE_MAX)
    (faibles if douteux else polys).append(g)
points, manques = [], []
for f in pts.getFeatures():
    p = tr.transform(f.geometry().asPoint())
    points.append(p)
    if f["emprise"] == 0:
        manques.append(p)

idx_pts = QgsSpatialIndex()
for i, p in enumerate(points):
    idx_pts.addFeature(i, QgsRectangle(p.x(), p.y(), p.x(), p.y()))


def index_geoms(geoms):
    idx = QgsSpatialIndex()
    for i, g in enumerate(geoms):
        idx.addFeature(i, g.boundingBox())
    return idx


idx_poly, idx_faible = index_geoms(polys), index_geoms(faibles)
idx_manq = QgsSpatialIndex()
for i, p in enumerate(manques):
    idx_manq.addFeature(i, QgsRectangle(p.x(), p.y(), p.x(), p.y()))

xs, ys = [p.x() for p in points], [p.y() for p in points]
centre = QgsPointXY(sum(xs) / len(xs), sum(ys) / len(ys))
upm = src.units_per_meter(centre)
step = RES * upm
cote = T * step
marge = 100 * upm
x_min, x_max, y_min, y_max = min(xs) - marge, max(xs) + marge, min(ys) - marge, max(ys) + marge


def anneaux_px(g, x0, y0):
    polys_ = g.asMultiPolygon() if g.isMultipart() else [g.asPolygon()]
    for poly in polys_:
        yield [np.array([[(q.x() - x0) / step, (y0 - q.y()) / step] for q in anneau], np.int32)
               for anneau in poly]


os.makedirs(sortie, exist_ok=True)
noyau = np.ones((3, 3), np.uint8)
n = 0
j = 0
y0 = y_max
while y0 - cote > y_min - cote:
    i = 0
    x0 = x_min
    while x0 < x_max:
        ext = QgsRectangle(x0, y0 - cote, x0 + cote, y0)
        c = ext.center()
        d = DIST_VILLAGE * upm
        proches = idx_pts.intersects(QgsRectangle(c.x() - d, c.y() - d, c.x() + d, c.y() + d))
        if proches:
            img = src.read(ext, T, T)
            if img.max() > 0 and (img.reshape(-1, 3).std(axis=0).mean() > 4):
                inst = np.zeros((T, T), np.int32)
                for k in idx_poly.intersects(ext):
                    for anneaux in anneaux_px(polys[k], x0, y0):
                        cv2.fillPoly(inst, [anneaux[0]], k + 1)
                        for trou in anneaux[1:]:
                            cv2.fillPoly(inst, [trou], 0)
                # emprises surtout vertes : arbres, pas des toits
                rgb = img.astype(np.int16)
                # houppiers : sombres et plus verts que rouges (les toits sont clairs)
                veg = (rgb[..., 1] > rgb[..., 0] + 5) & (rgb.sum(axis=2) < 260)
                nb = np.bincount(inst.ravel(), minlength=len(polys) + 1)
                nv = np.bincount(inst.ravel(), weights=veg.ravel(), minlength=len(polys) + 1)
                arbres = np.nonzero((nb > 0) & (nv > PART_VEGETATION_MAX * np.maximum(nb, 1)))[0]
                arbres = arbres[arbres > 0]
                masque_arbres = np.isin(inst, arbres)
                inst[masque_arbres] = 0
                bat = inst > 0
                # contour : pixels dont le voisinage contient une autre étiquette
                lab = inst.astype(np.float32)
                bord = (cv2.dilate(lab, noyau) != cv2.erode(lab, noyau))
                bord = cv2.dilate(bord.astype(np.uint8), noyau).astype(bool) & (
                    cv2.dilate(bat.astype(np.uint8), noyau).astype(bool))
                masque = np.zeros((T, T), np.uint8)
                masque[bat] = 1
                masque[bord] = 2
                ign = np.zeros((T, T), np.uint8)
                for k in idx_faible.intersects(ext):
                    for anneaux in anneaux_px(faibles[k], x0, y0):
                        cv2.fillPoly(ign, [anneaux[0]], 1)
                ign[masque_arbres] = 1
                ign = cv2.dilate(ign, noyau, iterations=3)
                r = int(RAYON_IGNORE / RES)
                for k in idx_manq.intersects(ext.buffered(RAYON_IGNORE * upm)):
                    p = manques[k]
                    cv2.circle(ign, (int((p.x() - x0) / step), int((y0 - p.y()) / step)), r, 1, -1)
                masque[(ign > 0) & (masque == 0)] = 255
                nom = f"{j:03d}_{i:03d}"
                cv2.imwrite(os.path.join(sortie, f"img_{nom}.png"), img[..., ::-1])
                cv2.imwrite(os.path.join(sortie, f"msk_{nom}.png"), masque)
                n += 1
        x0 += cote
        i += 1
    y0 -= cote
    j += 1
print(f"{gpkg} [{fond}] : {n} vignettes de {T} px", flush=True)
