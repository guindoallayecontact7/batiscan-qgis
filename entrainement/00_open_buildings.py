"""Extrait les emprises Google/Microsoft Open Buildings autour de chaque village.

python-qgis 00_open_buildings.py <dossier_sortie> <points1.shp> [points2.shp ...]

Source : VIDA « google-microsoft-open-buildings » (GeoParquet par pays, source.coop).
Licences : Google Open Buildings CC BY-4.0, Microsoft Building Footprints ODbL.
Ces emprises complètent les étiquettes issues des points : un relevé compte
souvent un point par concession, alors qu'une concession a plusieurs toits.
"""

import os
import re
import subprocess
import sys

from commun_qgis import demarrer_qgis

app = demarrer_qgis()

from qgis.core import (  # noqa: E402
    QgsCoordinateReferenceSystem,
    QgsCoordinateTransform,
    QgsProject,
    QgsVectorLayer,
)

URL = ("/vsicurl/https://data.source.coop/vida/google-microsoft-open-buildings/"
       "geoparquet/by_country/country_iso={pays}/{pays}.parquet")
sortie = sys.argv[1]
os.makedirs(sortie, exist_ok=True)
ogr2ogr = os.path.join(os.path.dirname(sys.executable), "ogr2ogr.exe")
if not os.path.exists(ogr2ogr):
    ogr2ogr = "ogr2ogr"
wgs84 = QgsCoordinateReferenceSystem("EPSG:4326")
for shp in sys.argv[2:]:
    pts = QgsVectorLayer(shp, "p", "ogr")
    ext = QgsCoordinateTransform(pts.crs(), wgs84, QgsProject.instance()).transformBoundingBox(pts.extent())
    ext = ext.buffered(0.003)  # ≈ 300 m
    nom = re.sub(r"[^A-Za-z0-9]", "_", os.path.basename(shp)[:-4].replace("Total_building ", ""))
    cible = os.path.join(sortie, nom + ".gpkg")
    cmd = [ogr2ogr, "-f", "GPKG", cible, URL.format(pays="MLI"), "-nln", "batiments", "-overwrite",
           "-spat", str(ext.xMinimum()), str(ext.yMinimum()), str(ext.xMaximum()), str(ext.yMaximum())]
    r = subprocess.run(cmd, capture_output=True, text=True)
    n = QgsVectorLayer(cible + "|layername=batiments", "b", "ogr").featureCount() if r.returncode == 0 else -1
    print(f"{nom} : {n} bâtiments Open Buildings ({r.stderr.strip()[-200:]})", flush=True)
