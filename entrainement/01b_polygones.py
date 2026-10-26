"""Étape 1 bis : emprises déjà détourées (polygones) au lieu de points de terrain.

python-qgis 01b_polygones.py <polygones.shp> <google|bing> <sortie.gpkg> [modele_v1.onnx] [conf_min=0.5]

Pour des polygones produits par un autre outil (par exemple le plugin
« AI Segmentation » de TerraLab, avec un champ « confidence »). Écrit le même
GeoPackage que 01_emprises_sam.py, que 02_jeu.py sait lire :
- « emprises » : les polygones ; fiabilité « faible » (donc ignorée à
  l'entraînement) sous conf_min ;
- « points » : le centre de chaque polygone (emprise = 1) et, si un modèle
  est fourni, le centre de chaque bâtiment qu'il détecte et que les polygones
  n'ont pas (emprise = 0). 02_jeu.py neutralise ces zones : un bâtiment
  oublié par les polygones, souvent une case ronde, n'est pas appris comme
  du fond.
"""

import sys

from commun_qgis import couche_fond, demarrer_qgis

app = demarrer_qgis()

from qgis.core import (  # noqa: E402
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsProject,
    QgsSpatialIndex,
    QgsVectorFileWriter,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QMetaType  # noqa: E402

from batiscan.core.image_source import SourceImage  # noqa: E402
from batiscan.core.tasks import TacheAuto, _mesureur, champs_emprises, couche_memoire  # noqa: E402

shp, fond, sortie = sys.argv[1:4]
modele = sys.argv[4] if len(sys.argv) > 4 else None
conf_min = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5

poly = QgsVectorLayer(shp, "polygones", "ogr")
if not poly.isValid() or poly.featureCount() == 0:
    sys.exit(f"Aucun polygone lu dans {shp}")
crs = poly.crs()
da = _mesureur(crs)
champ_conf = poly.fields().indexOf("confidence")

emprises, centres, geoms = [], [], []
for f in poly.getFeatures():
    if not f.hasGeometry():
        continue
    g = QgsGeometry(f.geometry())
    conf = float(f[champ_conf]) if champ_conf >= 0 and f[champ_conf] is not None else 1.0
    fiab = "haute" if conf >= conf_min + 0.1 else ("moyenne" if conf >= conf_min else "faible")
    # Un grand polygone qui ne ressemble pas à un rectangle est le plus souvent
    # une cour ou un terrain : forme « libre », que 02_jeu.py ignore au-delà de 250 m².
    rect = g.orientedMinimumBoundingBox()[0]
    forme = "rectangle" if rect.area() > 0 and g.area() / rect.area() >= 0.8 else "libre"
    e = QgsFeature(champs_emprises())
    e.setGeometry(g)
    e.setAttributes([f.id(), 1, "polygone", fiab, round(conf, 3), forme,
                     round(da.measureArea(g), 1), shp, ""])
    emprises.append(e)
    centres.append((g.pointOnSurface(), 1))
    geoms.append(g)

manques = 0
if modele:
    src = SourceImage(couche_fond(fond))
    tr = QgsCoordinateTransform(crs, src.crs, QgsProject.instance())
    ext = tr.transformBoundingBox(poly.extent())
    ext = ext.buffered(50 * src.units_per_meter(ext.center()))
    t = TacheAuto(src, ext, crs, modele, {"resolution": 0.3, "seuil": None, "aire_min": 4,
                                           "regularisation": "auto", "gpu": True})
    if not t.run():
        sys.exit(t.erreur or "échec de la détection")
    index = QgsSpatialIndex()
    for i, g in enumerate(geoms):
        index.addFeature(i, g.boundingBox())
    for d in t.resultats:
        g = d.geometry()
        if not any(geoms[i].intersects(g) for i in index.intersects(g.boundingBox())):
            centres.append((g.pointOnSurface(), 0))
            manques += 1

opts = QgsVectorFileWriter.SaveVectorOptions()
opts.driverName = "GPKG"
opts.layerName = "emprises"
QgsVectorFileWriter.writeAsVectorFormatV3(couche_memoire("emprises", crs, "Polygon", champs_emprises(), emprises),
                                          sortie, QgsProject.instance().transformContext(), opts)
champs = QgsFields()
champs.append(QgsField("emprise", QMetaType.Type.Int))
pts = []
for g, v in centres:
    p = QgsFeature(champs)
    p.setGeometry(g)
    p.setAttributes([v])
    pts.append(p)
opts.layerName = "points"
opts.actionOnExistingFile = QgsVectorFileWriter.ActionOnExistingFile.CreateOrOverwriteLayer
QgsVectorFileWriter.writeAsVectorFormatV3(couche_memoire("points", crs, "Point", champs, pts),
                                          sortie, QgsProject.instance().transformContext(), opts)
gardes = sum(1 for e in emprises if e["fiabilite"] != "faible")
print(f"{shp} : {len(emprises)} polygones ({gardes} gardés, confiance >= {conf_min}), "
      f"{manques} bâtiments vus par le modèle et absents des polygones (zones neutralisées)", flush=True)
