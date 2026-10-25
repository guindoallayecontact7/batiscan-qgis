"""Masques raster -> polygones QGIS, puis régularisation des formes.

Le bâti sahélien mêle toits rectangulaires (tôle, banco) et cases rondes :
le mode « auto » essaie un rectangle orienté et un cercle, et garde la forme
qui recouvre le mieux le masque, sinon un contour simplifié.
"""

import math

import cv2
import numpy as np
from qgis.core import Qgis, QgsGeometry, QgsPointXY


def mask_to_geometries(mask, geo, min_pixels=4):
    """Contours d'un masque booléen en géométries (SCR de l'image).

    geo = (x0, y0, pas_x, pas_y) : coin haut-gauche et taille de pixel.
    Chaque tache donne un polygone, trous compris.
    """
    x0, y0, sx, sy = geo
    m = np.ascontiguousarray(mask.astype(np.uint8))
    contours, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]

    def ring(cnt):
        pts = [QgsPointXY(x0 + (p[0][0] + 0.5) * sx, y0 - (p[0][1] + 0.5) * sy) for p in cnt]
        if len(pts) < 3:
            return None
        pts.append(pts[0])
        return pts

    geoms = []
    for i, cnt in enumerate(contours):
        if hier[i][3] != -1 or cv2.contourArea(cnt) < min_pixels:
            continue
        outer = ring(cnt)
        if outer is None:
            continue
        rings = [outer]
        child = hier[i][2]
        while child != -1:
            if cv2.contourArea(contours[child]) >= min_pixels:
                r = ring(contours[child])
                if r:
                    rings.append(r)
            child = hier[child][0]
        g = QgsGeometry.fromPolygonXY(rings)
        # Les contours passent par le centre des pixels de bord : on rend
        # le demi-pixel perdu pour conserver la surface.
        g = g.buffer(0.5 * sx, 2, Qgis.EndCapStyle.Flat, Qgis.JoinStyle.Miter, 2.0)
        if not g.isGeosValid():
            g = g.makeValid()
        if not g.isEmpty():
            geoms.append(g)
    return geoms


def _iou(a, b):
    inter = a.intersection(b).area()
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0


def regularize(geom, mode="auto", tolerance=0.3, min_iou=0.78):
    """Renvoie (géométrie, forme). mode : brut, simplifie, rectangle, auto.

    tolerance est en unités du SCR (≈ une taille de pixel).
    """
    if mode == "brut":
        return geom, "libre"
    simp = geom.simplify(tolerance)
    if simp.isEmpty():
        simp = geom
    if mode == "simplifie":
        return simp, "libre"
    rect = geom.orientedMinimumBoundingBox()[0]
    if mode == "rectangle":
        return rect, "rectangle"
    c = geom.centroid().asPoint()
    r = math.sqrt(max(geom.area(), 0) / math.pi)
    circle = QgsGeometry.fromPointXY(c).buffer(r, 8)
    i_rect, i_circ = _iou(geom, rect), _iou(geom, circle)
    if max(i_rect, i_circ) >= min_iou:
        return (rect, "rectangle") if i_rect >= i_circ else (circle, "cercle")
    return simp, "libre"
