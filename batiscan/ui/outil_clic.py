"""Outil carte : un clic sur un toit ajoute son emprise (SAM)."""

from datetime import date

from qgis.core import (
    Qgis,
    QgsCoordinateTransform,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsRectangle,
)
from qgis.gui import QgsMapToolEmitPoint
from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtWidgets import QApplication

from ..core.point_prompt import Criteres, segmenter_point
from ..core.tasks import FIABILITE, _mesureur, champs_emprises
from ..core.vectorize import mask_to_geometries, regularize


class OutilClic(QgsMapToolEmitPoint):
    """Clic gauche : ajoute le bâtiment ; clic droit : retire le dernier ajout."""

    def __init__(self, iface, contexte):
        super().__init__(iface.mapCanvas())
        self.iface = iface
        self.contexte = contexte  # callable -> dict(source, sam, couche, params) ou None
        self._vignette = None  # (cle, emprise du cœur, geo) de l'empreinte SAM en cache
        self._ajouts = []
        self.setCursor(Qt.CursorShape.CrossCursor)

    def canvasReleaseEvent(self, e):
        ctx = self.contexte()
        if not ctx:
            return
        couche = ctx["couche"]
        if e.button() == Qt.MouseButton.RightButton:
            if self._ajouts:
                couche.dataProvider().deleteFeatures([self._ajouts.pop()])
                couche.triggerRepaint()
            return
        src, sam, p = ctx["source"], ctx["sam"], ctx["params"]
        canvas_crs = self.canvas().mapSettings().destinationCrs()
        tr = QgsCoordinateTransform(canvas_crs, src.crs, QgsProject.instance())
        pt = tr.transform(self.toMapCoordinates(e.pos()))
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            T = p["vignette"]
            cle = (id(src), id(sam), T, p["resolution"])
            if (self._vignette is None or self._vignette[0] != cle
                    or not self._vignette[1].contains(pt)):
                img, geo = src.read_window(pt, T, p["resolution"])
                sam.set_image(img)
                x0, y0, sx, sy = geo
                q = T / 4 * sx
                coeur = QgsRectangle(x0 + q, y0 - T * sy + q, x0 + T * sx - q, y0 - q)
                self._vignette = (cle, coeur, geo)
            x0, y0, sx, sy = self._vignette[2]
            pc, pr = (pt.x() - x0) / sx, (y0 - pt.y()) / sy
            crit = Criteres(p["resolution"], p["aire_min"], p["aire_max"], rayon_recherche_m=1.5)
            res = segmenter_point(sam, pc, pr, [], crit)
            if res is None:
                self.iface.messageBar().pushMessage(
                    "BatiScan", "Pas de bâtiment net sous le clic. Cliquez au centre du toit.",
                    Qgis.MessageLevel.Warning, 3)
                return
            geoms = mask_to_geometries(res["masque"], self._vignette[2])
            if not geoms:
                return
            ptg = QgsGeometry.fromPointXY(QgsPointXY(pt))
            geom = min(geoms, key=lambda g: g.distance(ptg))
            geom, forme = regularize(geom, p["regularisation"], tolerance=sx)
            geom.transform(QgsCoordinateTransform(src.crs, couche.crs(), QgsProject.instance()))
            f = QgsFeature(champs_emprises())
            f.setGeometry(geom)
            surface = _mesureur(couche.crs()).measureArea(geom)
            f.setAttributes([None, 1, "interactif", FIABILITE["interactif"], round(res["score"], 3),
                             forme, round(surface, 1), src.name, date.today().isoformat()])
            ok, ajout = couche.dataProvider().addFeatures([f])
            if ok and ajout:
                self._ajouts.append(ajout[0].id())
            couche.updateExtents()
            couche.triggerRepaint()
        except Exception as exc:  # l'outil ne doit pas planter QGIS
            self.iface.messageBar().pushMessage("BatiScan", str(exc), Qgis.MessageLevel.Critical, 6)
        finally:
            QApplication.restoreOverrideCursor()

    def oublier(self):
        self._vignette = None
