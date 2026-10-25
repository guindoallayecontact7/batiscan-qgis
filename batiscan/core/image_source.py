"""Lecture de fenêtres RVB depuis n'importe quelle couche raster QGIS.

Fonctionne avec les GeoTIFF/orthophotos (1 à n bandes, 8 ou 16 bits) et avec les
fonds XYZ/WMS (Google, Esri, Bing...) que le fournisseur renvoie en ARGB32.
La lecture passe par le fournisseur de données cloné : elle peut donc tourner
dans une QgsTask, hors du fil principal.
"""

import numpy as np
from qgis.core import (
    Qgis,
    QgsDistanceArea,
    QgsPointXY,
    QgsProject,
    QgsRectangle,
)

_DTYPES = {
    Qgis.DataType.Byte: np.uint8,
    Qgis.DataType.UInt16: np.uint16,
    Qgis.DataType.Int16: np.int16,
    Qgis.DataType.UInt32: np.uint32,
    Qgis.DataType.Int32: np.int32,
    Qgis.DataType.Float32: np.float32,
    Qgis.DataType.Float64: np.float64,
}
_ARGB = (Qgis.DataType.ARGB32, Qgis.DataType.ARGB32_Premultiplied)


class SourceImage:
    """Fenêtres RVB uint8 d'une couche raster, dans le SCR de la couche.

    À construire dans le fil principal (clonage du fournisseur, statistiques),
    puis utilisable depuis une tâche de fond.
    """

    def __init__(self, layer, bands=(1, 2, 3)):
        self.name = layer.name()
        self.crs = layer.crs()
        self.extent = QgsRectangle(layer.extent())
        self.provider = layer.dataProvider().clone()
        self.is_argb = self.provider.dataType(1) in _ARGB
        nb = self.provider.bandCount()
        if self.is_argb:
            self.bands = (1,)
        elif nb >= 3:
            self.bands = tuple(b for b in bands if b <= nb)[:3]
        else:
            self.bands = (1, 1, 1)
        # Étirement pour les images non 8 bits (2 % - 98 % sur la couche entière)
        self.ranges = {}
        if not self.is_argb:
            for b in set(self.bands):
                if self.provider.dataType(b) != Qgis.DataType.Byte:
                    lo, hi = self.provider.cumulativeCut(b, 0.02, 0.98, self.extent, 250000)
                    self.ranges[b] = (lo, hi if hi > lo else lo + 1)
        self._da = QgsDistanceArea()
        self._da.setSourceCrs(self.crs, QgsProject.instance().transformContext())
        self._da.setEllipsoid("EPSG:7030")

    # -- unités -----------------------------------------------------------
    def units_per_meter(self, near):
        """Nombre d'unités du SCR pour un mètre au sol, autour du point `near`.

        Vaut 1 en UTM, environ 1/cos(lat) en Web Mercator (EPSG:3857),
        environ 1/111 320 en degrés. Le point compte : l'emprise d'une couche
        XYZ couvre le monde entier, son centre ne dit rien de la zone traitée.
        """
        d = 1e-4 if self.crs.isGeographic() else 10.0
        p1 = QgsPointXY(near.x(), near.y())
        p2 = QgsPointXY(near.x() + d, near.y())
        metres = self._da.measureLine(p1, p2)
        return d / metres if metres > 0 else 1.0

    # -- lecture ----------------------------------------------------------
    def read(self, extent, width, height):
        """Renvoie un tableau (height, width, 3) uint8 pour l'emprise donnée."""
        width, height = int(width), int(height)
        if self.is_argb:
            blk = self.provider.block(1, extent, width, height)
            buf = np.frombuffer(bytes(blk.data()), dtype=np.uint8)
            if buf.size != width * height * 4:
                return np.zeros((height, width, 3), np.uint8)
            bgra = buf.reshape(height, width, 4)
            return np.ascontiguousarray(bgra[..., [2, 1, 0]])
        canaux = []
        cache = {}
        for b in self.bands:
            if b not in cache:
                blk = self.provider.block(b, extent, width, height)
                dt = _DTYPES.get(blk.dataType(), np.float32)
                arr = np.frombuffer(bytes(blk.data()), dtype=dt)
                if arr.size != width * height:
                    arr = np.zeros(width * height, dt)
                arr = arr.reshape(height, width)
                if b in self.ranges:
                    lo, hi = self.ranges[b]
                    arr = np.clip((arr.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255)
                cache[b] = arr.astype(np.uint8)
            canaux.append(cache[b])
        return np.dstack(canaux)

    def read_window(self, center, size_px, res_m):
        """Fenêtre carrée de `size_px` pixels de `res_m` mètres centrée sur `center`.

        Renvoie (image, geo) où geo = (x0, y0, pas_x, pas_y) : coin haut-gauche
        et taille de pixel en unités du SCR.
        """
        step = res_m * self.units_per_meter(center)
        half = size_px * step / 2.0
        ext = QgsRectangle(center.x() - half, center.y() - half,
                           center.x() + half, center.y() + half)
        return self.read(ext, size_px, size_px), (ext.xMinimum(), ext.yMaximum(), step, step)
