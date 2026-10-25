"""Étape 1 : emprises SAM à partir des points de terrain d'un village.

python-qgis 01_emprises_sam.py <points.shp> <google|bing> <sortie.gpkg>

Écrit deux couches : « emprises » (polygones, champs BatiScan) et « points »
(tous les points, avec un champ « emprise » = 1 si un polygone a été trouvé).
"""

import sys
import time

from commun_qgis import couche_fond, demarrer_qgis

app = demarrer_qgis()

from qgis.core import (  # noqa: E402
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsProject,
    QgsVectorFileWriter,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QMetaType  # noqa: E402

from batiscan.core import models  # noqa: E402
from batiscan.core.image_source import SourceImage  # noqa: E402
from batiscan.core.tasks import TachePoints, champs_emprises, couche_memoire  # noqa: E402

shp, fond, sortie = sys.argv[1:4]
pts = QgsVectorLayer(shp, "points", "ogr")
src = SourceImage(couche_fond(fond))
tr = QgsCoordinateTransform(pts.crs(), src.crs, QgsProject.instance())
points = [(f.id(), tr.transform(f.geometry().asPoint())) for f in pts.getFeatures() if f.hasGeometry()]
if not points:
    sys.exit(f"Aucun point lu dans {shp} (couche valide : {pts.isValid()})")
sam = models.chemins(next(m for m in models.CATALOGUE if m["id"] == "mobilesam"))
if sam is None:
    sys.exit("MobileSAM n'est pas installé dans le profil QGIS")
params = {"resolution": 0.3, "vignette": 192, "rayon_voisins": 15, "rayon_recherche": 3,
          "aire_min": 4, "aire_max": 600, "regularisation": "auto", "gpu": False}
t0 = time.time()
tache = TachePoints(src, points, pts.crs(), sam, params)
if not tache.run():
    sys.exit(tache.erreur or "échec")
print(f"{shp}: {len(points)} points, {len(tache.resultats)} emprises, {len(tache.echecs)} échecs, "
      f"{dict(tache.stats)}, {time.time() - t0:.0f} s", flush=True)

opts = QgsVectorFileWriter.SaveVectorOptions()
opts.driverName = "GPKG"
opts.layerName = "emprises"
lyr = couche_memoire("emprises", pts.crs(), "Polygon", champs_emprises(), tache.resultats)
QgsVectorFileWriter.writeAsVectorFormatV3(lyr, sortie, QgsProject.instance().transformContext(), opts)

echecs = set(tache.echecs)
champs = QgsFields()
champs.append(QgsField("emprise", QMetaType.Type.Int))
feats = []
for f in pts.getFeatures():
    g = QgsFeature(champs)
    g.setGeometry(f.geometry())
    g.setAttributes([0 if f.id() in echecs else 1])
    feats.append(g)
opts.layerName = "points"
opts.actionOnExistingFile = QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteLayer
QgsVectorFileWriter.writeAsVectorFormatV3(couche_memoire("points", pts.crs(), "Point", champs, feats),
                                          sortie, QgsProject.instance().transformContext(), opts)
