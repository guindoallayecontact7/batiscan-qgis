"""Étape 5 : détection automatique sur un village, sans ses points, puis comparaison aux points.

python-qgis 05_evaluer.py <modele.onnx> <points.shp> <google|bing> <sortie.gpkg> [res=0.3]

La détection couvre l'emprise des points (plus 50 m). Les points ne servent
qu'à mesurer : part des points couverts par une emprise détectée (rappel)
et part des emprises qui contiennent un point (précision approchée, car un
bâtiment non relevé compte comme une fausse détection).
"""

import sys
import time

from commun_qgis import couche_fond, demarrer_qgis

app = demarrer_qgis()

from qgis.core import (  # noqa: E402
    QgsCoordinateTransform,
    QgsProject,
    QgsVectorFileWriter,
    QgsVectorLayer,
)

from batiscan.core import qc  # noqa: E402
from batiscan.core.image_source import SourceImage  # noqa: E402
from batiscan.core.tasks import TacheAuto, champs_emprises, couche_memoire  # noqa: E402

onnx_path, shp, fond, sortie = sys.argv[1:5]
res = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
pts = QgsVectorLayer(shp, "points", "ogr")
src = SourceImage(couche_fond(fond))
tr = QgsCoordinateTransform(pts.crs(), src.crs, QgsProject.instance())
ext = tr.transformBoundingBox(pts.extent())
ext = ext.buffered(50 * src.units_per_meter(ext.center()))
t0 = time.time()
tache = TacheAuto(src, ext, pts.crs(), onnx_path,
                  {"resolution": res, "seuil": None, "aire_min": 4, "regularisation": "auto", "gpu": True})
if not tache.run():
    sys.exit(tache.erreur or "échec")
duree = time.time() - t0
lyr = couche_memoire("detection", pts.crs(), "Polygon", champs_emprises(), tache.resultats)
r, _ = qc.controler(pts, lyr, 1.0)
precision = r["emprises_avec_point"] / max(r["emprises"], 1)
print(f"{shp} [{fond}] : {len(tache.resultats)} emprises en {duree:.0f} s ; "
      f"rappel {100 * r['taux_couverture']:.1f} % ({r['points_couverts']}/{r['points']}) ; "
      f"précision approchée {100 * precision:.1f} % ; fusions {r['emprises_multi_points']}", flush=True)
opts = QgsVectorFileWriter.SaveVectorOptions()
opts.driverName = "GPKG"
opts.layerName = "detection"
QgsVectorFileWriter.writeAsVectorFormatV3(lyr, sortie, QgsProject.instance().transformContext(), opts)
