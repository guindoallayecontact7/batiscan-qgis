"""Outils partagés par les scripts d'entraînement qui tournent avec le Python de QGIS."""

import os
import sys
from urllib.parse import quote

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FONDS = {
    "google": "https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
    "bing": "https://ecn.t3.tiles.virtualearth.net/tiles/a{q}.jpeg?g=1",
    "esri": "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
}
ZMAX = {"google": 20, "bing": 19, "esri": 17}  # Esri : pas de zoom 18+ sur la plupart du Mali


def demarrer_qgis():
    from qgis.core import QgsApplication

    app = QgsApplication([], False)
    app.initQgis()
    return app


def couche_fond(nom):
    from qgis.core import QgsRasterLayer

    url = quote(FONDS[nom], safe=":/?")
    uri = f"type=xyz&url={url}&zmax={ZMAX[nom]}&zmin=0"
    lyr = QgsRasterLayer(uri, nom, "wms")
    if not lyr.isValid():
        raise RuntimeError("Fond invalide : " + nom)
    return lyr
