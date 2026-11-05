"""Formes finales de la détection automatique : un objet par bâtiment, sans superposition.

Les géométries arrivent brutes, dans le SCR de l'image. Les réglages sont en mètres ;
upm convertit en unités du SCR. Quatre étapes : retrait des murs et appendices fins,
fusion des morceaux d'un même bâtiment, régularisation, suppression des chevauchements.
"""

import math

import cv2
import numpy as np
from qgis.core import Qgis, QgsGeometry, QgsPointXY, QgsRectangle, QgsSpatialIndex

LARGEUR_MIN = 1.8  # m : plus fin, c'est un mur ou un appendice
AIRE_MIN = 8.0  # m²
ECART_FUSION = 1.0  # m
AIRE_FUSION_MAX = 400.0  # m²
PIXEL = 0.3  # m : grille du tracé à angles droits
SEUIL_FORME = 0.6  # surface / rectangle orienté, sous ce seuil la tache est rejetée
SEUIL_CONVEXE = 0.7  # surface / enveloppe convexe
SEUIL_CONTRASTE = 20.0  # écart Lab avec le sol autour : en dessous, cour et toit en terre se confondent


def _tampon(g, d):
    return g.buffer(d, 2, Qgis.EndCapStyle.Flat, Qgis.JoinStyle.Miter, 2.0)


def _plus_grand(g):
    """Plus grande partie polygonale, ou None."""
    if g is None or g.isEmpty():
        return None
    parts = [p for p in g.asGeometryCollection() if p.type() == Qgis.GeometryType.Polygon and p.area() > 0]
    return max(parts, key=lambda p: p.area()) if parts else None


def _boite(g):
    return g.orientedMinimumBoundingBox()[0]


def _droit(g):
    return g.area() / max(_boite(g).area(), 1e-9)


def _anneau(g):
    return np.array([(p.x(), p.y()) for p in g.asPolygon()[0]], float)


def _polygone(xy):
    return QgsGeometry.fromPolygonXY([[QgsPointXY(float(x), float(y)) for x, y in xy]])


def _tourner(xy, t, c):
    co, si = math.cos(t), math.sin(t)
    d = xy - c
    return np.c_[d[:, 0] * co - d[:, 1] * si, d[:, 0] * si + d[:, 1] * co] + c


def _centre(g):
    c = g.centroid().asPoint()
    return np.array([c.x(), c.y()])


def _rectangle(g):
    """Rectangle orienté ramené à la surface de la tache."""
    b = _boite(g)
    f = math.sqrt(g.area() / max(b.area(), 1e-9))
    c = _centre(g)
    return _polygone(c + (_anneau(b) - c) * f)


def _orthogonal(g, upm):
    """Polygone à angles droits, dans l'axe du bâtiment."""
    b = _anneau(_boite(g))
    a = max(((b[i], b[i + 1]) for i in range(2)), key=lambda s: np.hypot(*(s[1] - s[0])))
    t = math.atan2(a[1][1] - a[0][1], a[1][0] - a[0][0])
    c = _centre(g)
    r = _tourner(_anneau(g), -t, c)
    pas = PIXEL * upm
    o = r.min(axis=0)
    w, h = ((r.max(axis=0) - o) / pas).astype(int) + 9
    m = np.zeros((h, w), np.uint8)
    cv2.fillPoly(m, [((r - o) / pas + 4).astype(np.int32)], 1)
    k = np.ones((5, 5), np.uint8)
    m = cv2.morphologyEx(cv2.morphologyEx(m, cv2.MORPH_CLOSE, k), cv2.MORPH_OPEN, k)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return _rectangle(g)
    q = max(cs, key=cv2.contourArea)[:, 0, :].astype(float)
    if len(q) < 4:
        return _rectangle(g)
    q = np.vstack([q, q[:1]])
    return _plus_grand(_polygone(_tourner((q - 4 + 0.5) * pas + o, t, c)).makeValid())


def regulariser(g, upm):
    """Renvoie (géométrie, forme), ou (None, « rejet ») pour une tache qui n'est pas un bâtiment."""
    d = _droit(g)
    circ = 4 * math.pi * g.area() / max(g.length(), 1e-9) ** 2
    if circ >= 0.86 and d < 0.84 and g.area() / upm ** 2 < 80:
        return QgsGeometry.fromPointXY(g.centroid().asPoint()).buffer(math.sqrt(g.area() / math.pi), 8), "cercle"
    if d >= 0.72:
        return _rectangle(g), "rectangle"
    o = _orthogonal(g, upm)
    if o is None or _droit(o) < SEUIL_FORME or g.area() / max(g.convexHull().area(), 1e-9) < SEUIL_CONVEXE:
        return None, "rejet"
    return o, "angles_droits"


def _index(geoms):
    idx = QgsSpatialIndex()
    for i, g in enumerate(geoms):
        idx.addFeature(i, g.boundingBox())
    return idx


def _fusionner(geoms, upm):
    """Fusionne deux objets voisins si l'ensemble ressemble encore à un seul bâtiment."""
    gap = ECART_FUSION * upm
    idx = _index(geoms)
    paires = []
    for i, g in enumerate(geoms):
        r = QgsRectangle(g.boundingBox())
        r.grow(gap)
        for j in idx.intersects(r):
            if i < j:
                d = g.distance(geoms[j])
                if d <= gap:
                    paires.append((d, i, j))
    paires.sort()
    pere = list(range(len(geoms)))
    geoms = list(geoms)

    def racine(a):
        while pere[a] != a:
            pere[a] = pere[pere[a]]
            a = pere[a]
        return a

    e = gap / 2 + 0.05 * upm
    for _, a, b in paires:
        a, b = racine(a), racine(b)
        if a == b:
            continue
        ga, gb = geoms[a], geoms[b]
        u = _plus_grand(_tampon(_tampon(ga, e).combine(_tampon(gb, e)), -e))
        if u is None:
            continue
        ref = (_droit(ga) * ga.area() + _droit(gb) * gb.area()) / (ga.area() + gb.area())
        petit = min(ga.area(), gb.area()) / max(ga.area(), gb.area())
        du = _droit(u)
        if du >= max(0.8, ref - 0.03) and u.area() / upm ** 2 < AIRE_FUSION_MAX and (petit < 0.6 or du >= 0.88):
            geoms[a] = u
            pere[b] = a
    return [g for i, g in enumerate(geoms) if pere[i] == i]


def _sans_chevauchement(items, upm):
    """Les objets les plus sûrs et les plus grands passent d'abord ; les suivants sont coupés."""
    amin = AIRE_MIN * upm ** 2
    items = sorted(items, key=lambda x: -x[2] * math.sqrt(x[0].area()))
    idx = QgsSpatialIndex()
    gardes = []
    for g, forme, score in items:
        voisins = [gardes[i][0] for i in idx.intersects(g.boundingBox()) if gardes[i][0].intersects(g)]
        if voisins:
            r = _plus_grand(g.difference(QgsGeometry.unaryUnion(voisins).buffer(0.05 * upm, 2)))
            if r is None or r.area() < 0.6 * g.area() or r.area() < amin:
                continue
            g = r
        idx.addFeature(len(gardes), g.boundingBox())
        gardes.append((g, forme, score))
    return gardes


def _ecart(lab, z):
    """Écart de couleur Lab entre le cœur de l'objet et un anneau de 0,6 à 2,7 m."""
    def k(n):
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * n + 1, 2 * n + 1))

    coeur = cv2.erode(z, k(2))
    coeur = coeur if coeur.sum() >= 20 else z
    anneau = (cv2.dilate(z, k(9)) - cv2.dilate(z, k(2))) > 0
    if not coeur.any() or not anneau.any():
        return None
    return float(np.linalg.norm(np.median(lab[coeur > 0], 0) - np.median(lab[anneau], 0)))


def contrastes(source, geoms, upm):
    """Contraste de chaque objet avec son entourage (None si incalculable).

    L'image se lit par paquets d'objets voisins, pas objet par objet : un village
    compte des milliers d'objets et chaque lecture passe par le fournisseur raster.
    """
    pas = PIXEL * upm
    groupes = {}
    for i, g in enumerate(geoms):
        c = g.centroid().asPoint()
        groupes.setdefault((int(c.x() // (1024 * pas)), int(c.y() // (1024 * pas))), []).append(i)
    sortie = [None] * len(geoms)
    marge = 12
    for ids in groupes.values():
        bb = QgsRectangle(geoms[ids[0]].boundingBox())
        for i in ids:
            bb.combineExtentWith(geoms[i].boundingBox())
        x0, y1 = bb.xMinimum() - marge * pas, bb.yMaximum() + marge * pas
        w = int(math.ceil(bb.width() / pas)) + 2 * marge
        h = int(math.ceil(bb.height() / pas)) + 2 * marge
        img = source.read(QgsRectangle(x0, y1 - h * pas, x0 + w * pas, y1), w, h)
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB).astype(np.float32)
        for i in ids:
            p = _plus_grand(geoms[i])
            if p is None:
                continue
            xy = _anneau(p)
            px = np.c_[(xy[:, 0] - x0) / pas, (y1 - xy[:, 1]) / pas].astype(np.int32)
            c0, r0 = np.maximum(px.min(axis=0) - marge, 0)
            c1, r1 = px.max(axis=0) + marge + 1
            z = np.zeros((min(r1, h) - r0, min(c1, w) - c0), np.uint8)
            cv2.fillPoly(z, [px - (c0, r0)], 1)
            if z.any():
                sortie[i] = _ecart(lab[r0:r0 + z.shape[0], c0:c0 + z.shape[1]], z)
    return sortie


def construire(items, upm):
    """items : (géométrie brute, score). Renvoie une liste (géométrie, forme, score)."""
    amin = AIRE_MIN * upm ** 2
    demi = LARGEUR_MIN / 2 * upm
    geoms, scores = [], []
    for g, s in items:
        for p in g.asGeometryCollection():
            q = _plus_grand(_tampon(p.buffer(-demi, 2), demi))
            if q is not None and q.area() >= amin:
                geoms.append(q)
                scores.append(s)
    if not geoms:
        return []
    idx = _index(geoms)
    sortie = []
    for f in _fusionner(geoms, upm):
        ids = [i for i in idx.intersects(f.boundingBox()) if geoms[i].intersects(f)]
        s = float(np.mean([scores[i] for i in ids])) if ids else 0.5
        r, forme = regulariser(f, upm)
        if r is not None and r.area() >= amin:
            sortie.append((r, forme, s))
    return _sans_chevauchement(sortie, upm)
