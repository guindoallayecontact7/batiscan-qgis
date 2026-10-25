"""Traitements longs exécutés en tâche de fond (QgsTask).

Chaque tâche reçoit tout ce dont elle a besoin dans le fil principal
(SourceImage déjà construite, points déjà lus) et ne touche à aucune couche du
projet pendant run(). Les couches résultats sont créées dans finished().
"""

import math
import os
import traceback
from collections import defaultdict
from datetime import date

import cv2
import numpy as np
from qgis.core import (
    QgsCoordinateTransform,
    QgsDistanceArea,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsRectangle,
    QgsSpatialIndex,
    QgsTask,
    QgsVectorLayer,
)
from qgis.PyQt.QtCore import QMetaType, pyqtSignal

from .point_prompt import Criteres, segmenter_point
from .sam_onnx import SamOnnx
from .segmenter import OnnxSegmenter
from .vectorize import etiqueter, instances, mask_to_geometries, regularize

FIABILITE = {"direct": "haute", "coins": "moyenne", "recherche": "faible", "interactif": "haute"}


def champs_emprises():
    f = QgsFields()
    f.append(QgsField("id_point", QMetaType.Type.LongLong))
    f.append(QgsField("nb_points", QMetaType.Type.Int))
    f.append(QgsField("methode", QMetaType.Type.QString))
    f.append(QgsField("fiabilite", QMetaType.Type.QString))
    f.append(QgsField("score", QMetaType.Type.Double))
    f.append(QgsField("forme", QMetaType.Type.QString))
    f.append(QgsField("surface_m2", QMetaType.Type.Double))
    f.append(QgsField("source", QMetaType.Type.QString))
    f.append(QgsField("date", QMetaType.Type.QString))
    return f


def couche_memoire(nom, crs, geom="Polygon", champs=None, features=()):
    lyr = QgsVectorLayer(f"{geom}?crs={crs.authid() or 'EPSG:4326'}", nom, "memory")
    if not crs.authid():
        lyr.setCrs(crs)
    pr = lyr.dataProvider()
    if champs is not None:
        pr.addAttributes(champs.toList())
        lyr.updateFields()
    pr.addFeatures(list(features))
    lyr.updateExtents()
    return lyr


def _mesureur(crs):
    da = QgsDistanceArea()
    da.setSourceCrs(crs, QgsProject.instance().transformContext())
    da.setEllipsoid("EPSG:7030")
    return da


class _Dedoublonneur:
    """Fusionne les emprises quasi identiques produites par deux points."""

    def __init__(self, seuil=0.5):
        self.index = QgsSpatialIndex()
        self.feats = {}
        self.seuil = seuil
        self._n = 0

    def ajouter(self, feat):
        g = feat.geometry()
        for fid in self.index.intersects(g.boundingBox()):
            autre = self.feats[fid]
            ga = autre.geometry()
            inter = g.intersection(ga).area()
            if inter > self.seuil * min(g.area(), ga.area()):
                autre["nb_points"] = autre["nb_points"] + 1
                return False
        self._n += 1
        feat.setId(self._n)
        self.feats[self._n] = feat
        self.index.addFeature(feat)
        return True

    def features(self):
        return list(self.feats.values())


class _TacheBase(QgsTask):
    termine = pyqtSignal(object)  # émis dans le fil principal avec self

    def __init__(self, titre):
        super().__init__(titre, QgsTask.Flag.CanCancel)
        self.erreur = None
        self.resultats = []

    def finished(self, ok):
        self.termine.emit(self)


# --------------------------------------------------------------------------
class TachePoints(_TacheBase):
    """Une emprise par point de terrain, avec SAM."""

    def __init__(self, source, points, crs_sortie, chemins_sam, params):
        super().__init__("BatiScan : points → emprises")
        self.source = source
        self.points = points  # [(fid, QgsPointXY dans le SCR du raster)]
        self.crs_sortie = crs_sortie
        self.tr = QgsCoordinateTransform(source.crs, crs_sortie, QgsProject.instance().transformContext())
        self.sam_paths = chemins_sam
        self.p = params
        xs = [p.x() for _, p in points]
        ys = [p.y() for _, p in points]
        self.upm = source.units_per_meter(QgsPointXY(sum(xs) / len(xs), sum(ys) / len(ys)))
        self.echecs = []
        self.stats = defaultdict(int)

    def run(self):
        try:
            return self._run()
        except Exception:
            self.erreur = traceback.format_exc()
            return False

    def _run(self):
        p = self.p
        sam = SamOnnx(self.sam_paths["encodeur"], self.sam_paths["decodeur"],
                      self.sam_paths.get("canevas", (682, 1024)), p["gpu"])
        crit = Criteres(p["resolution"], p["aire_min"], p["aire_max"],
                        rayon_recherche_m=p["rayon_recherche"])
        T = p["vignette"]
        step = p["resolution"] * self.upm
        cell = (T // 2) * step  # cœur de vignette ; le reste sert de contexte
        r_vois = p["rayon_voisins"] * self.upm
        r_min = 3.0 * self.upm

        cells = defaultdict(list)
        grille = defaultdict(list)
        g = max(r_vois, step)
        for i, (_, pt) in enumerate(self.points):
            cells[(math.floor(pt.x() / cell), math.floor(pt.y() / cell))].append(i)
            grille[(math.floor(pt.x() / g), math.floor(pt.y() / g))].append(i)

        da = _mesureur(self.crs_sortie)
        dedup = _Dedoublonneur()
        aujourdhui = date.today().isoformat()
        total, fait = len(self.points), 0
        for (cx, cy), membres in sorted(cells.items()):
            if self.isCanceled():
                return False
            centre = QgsPointXY((cx + 0.5) * cell, (cy + 0.5) * cell)
            img, geo = self.source.read_window(centre, T, p["resolution"])
            if img.max() == 0:
                for i in membres:
                    self.echecs.append(self.points[i][0])
                    self.stats["sans_image"] += 1
                fait += len(membres)
                continue
            sam.set_image(img)
            x0, y0, sx, sy = geo
            for i in membres:
                fid, pt = self.points[i]
                pc, pr = (pt.x() - x0) / sx, (y0 - pt.y()) / sy
                vois = []
                gx, gy = math.floor(pt.x() / g), math.floor(pt.y() / g)
                for dx in (-1, 0, 1):
                    for dy in (-1, 0, 1):
                        for j in grille.get((gx + dx, gy + dy), ()):
                            q = self.points[j][1]
                            d = math.hypot(q.x() - pt.x(), q.y() - pt.y())
                            if j != i and r_min < d <= r_vois:
                                qc, qr = (q.x() - x0) / sx, (y0 - q.y()) / sy
                                if 0 <= qc < T and 0 <= qr < T:
                                    vois.append((qc, qr))
                res = segmenter_point(sam, pc, pr, vois, crit)
                fait += 1
                if res is None:
                    self.echecs.append(fid)
                    self.stats["echec"] += 1
                    continue
                geoms = mask_to_geometries(res["masque"], geo)
                if not geoms:
                    self.echecs.append(fid)
                    self.stats["echec"] += 1
                    continue
                ptg = QgsGeometry.fromPointXY(pt)
                geom = min(geoms, key=lambda gg: gg.distance(ptg))
                geom, forme = regularize(geom, p["regularisation"], tolerance=sx)
                geom.transform(self.tr)
                f = QgsFeature(champs_emprises())
                f.setGeometry(geom)
                f.setAttributes([fid, 1, res["methode"], FIABILITE[res["methode"]], round(res["score"], 3),
                                 forme, round(da.measureArea(geom), 1), self.source.name, aujourdhui])
                if dedup.ajouter(f):
                    self.stats[res["methode"]] += 1
                else:
                    self.stats["doublon"] += 1
            self.setProgress(100.0 * fait / total)
        self.resultats = dedup.features()
        return True


# --------------------------------------------------------------------------
class TacheAuto(_TacheBase):
    """Détection automatique sur une emprise avec un modèle de segmentation."""

    BLOC = 2048

    def __init__(self, source, emprise, crs_sortie, chemin_onnx, params):
        super().__init__("BatiScan : détection automatique")
        self.source = source
        self.emprise = QgsRectangle(emprise)  # SCR du raster
        self.crs_sortie = crs_sortie
        self.tr = QgsCoordinateTransform(source.crs, crs_sortie, QgsProject.instance().transformContext())
        self.chemin = chemin_onnx
        self.p = params
        self.upm = source.units_per_meter(self.emprise.center())
        self.resolution = None

    def run(self):
        try:
            return self._run()
        except Exception:
            self.erreur = traceback.format_exc()
            return False

    def _run(self):
        p = self.p
        seg = OnnxSegmenter(self.chemin, p["gpu"])
        res = p["resolution"] or seg.resolution_m
        self.resolution = res
        seuil = p["seuil"] if p["seuil"] is not None else seg.threshold
        step = res * self.upm
        W = max(1, math.ceil(self.emprise.width() / step))
        H = max(1, math.ceil(self.emprise.height() / step))
        marge = seg.tile // 2
        amin = p["aire_min"] / (res * res)
        da = _mesureur(self.crs_sortie)
        aujourdhui = date.today().isoformat()
        blocs = [(bx, by) for by in range(0, H, self.BLOC) for bx in range(0, W, self.BLOC)]
        feats = []
        for nb, (bx, by) in enumerate(blocs):
            if self.isCanceled():
                return False
            bw, bh = min(self.BLOC, W - bx), min(self.BLOC, H - by)
            w, h = bw + 2 * marge, bh + 2 * marge
            x0 = self.emprise.xMinimum() + (bx - marge) * step
            y0 = self.emprise.yMaximum() - (by - marge) * step
            ext = QgsRectangle(x0, y0 - h * step, x0 + w * step, y0)
            img = self.source.read(ext, w, h)

            def prog(v, nb=nb):
                self.setProgress((nb + v / 100.0) * 100.0 / len(blocs))

            prob, interieur = seg.probabilites_image(img, prog, self.isCanceled)
            lab = etiqueter(prob, interieur, seuil)
            for k, rr, cc in instances(lab):
                if len(rr) < amin:
                    continue
                if not (marge <= cc.mean() < marge + bw and marge <= rr.mean() < marge + bh):
                    continue  # traité par le bloc voisin
                t, l = int(rr.min()), int(cc.min())
                hh, ww = int(rr.max()) - t + 1, int(cc.max()) - l + 1
                sub = np.zeros((hh + 2, ww + 2), bool)
                sub[rr - t + 1, cc - l + 1] = True
                geo = (x0 + (l - 1) * step, y0 - (t - 1) * step, step, step)
                proba = float(prob[rr, cc].mean())
                for geom in mask_to_geometries(sub, geo):
                    geom, forme = regularize(geom, p["regularisation"], tolerance=step)
                    geom.transform(self.tr)
                    f = QgsFeature(champs_emprises())
                    f.setGeometry(geom)
                    f.setAttributes([None, 0, "auto", "haute" if proba >= 0.6 else "moyenne",
                                     round(proba, 3), forme, round(da.measureArea(geom), 1),
                                     self.source.name, aujourdhui])
                    feats.append(f)
        self.resultats = _sans_enceintes(feats)
        return True


def _sans_enceintes(feats, min_inclus=2):
    """Retire les formes qui englobent plusieurs autres détections.

    Le modèle trace parfois le mur d'enceinte d'une concession en plus des
    toits qu'elle contient : cette grande forme n'est pas un bâtiment.
    """
    index = QgsSpatialIndex()
    for i, f in enumerate(feats):
        f.setId(i)
        index.addFeature(f)
    centres = [f.geometry().pointOnSurface() for f in feats]
    garder = []
    for i, f in enumerate(feats):
        g = f.geometry()
        inclus = sum(1 for j in index.intersects(g.boundingBox())
                     if j != i and feats[j].geometry().area() < 0.5 * g.area() and g.contains(centres[j]))
        if inclus < min_inclus:
            garder.append(f)
    return garder


# --------------------------------------------------------------------------
class TacheJeuEntrainement(_TacheBase):
    """Vignettes image + masque (GeoTIFF) pour entraîner un modèle.

    Format compatible avec les tutoriels d'entraînement Deepness et la plupart
    des bibliothèques de segmentation : images/xxx.tif (RVB) et masks/xxx.tif
    (0 = fond, 1 = bâtiment).
    """

    def __init__(self, source, polygones, dossier, params):
        super().__init__("BatiScan : export du jeu d'entraînement")
        self.source = source
        self.polygones = polygones  # QgsGeometry dans le SCR du raster
        self.dossier = dossier
        self.p = params
        bb = QgsRectangle()
        for g in polygones:
            bb.combineExtentWith(g.boundingBox())
        self.bb = bb
        self.upm = source.units_per_meter(bb.center())
        self.n = 0

    def run(self):
        try:
            return self._run()
        except Exception:
            self.erreur = traceback.format_exc()
            return False

    def _run(self):
        from osgeo import gdal

        T, res = self.p["taille"], self.p["resolution"]
        step = res * self.upm
        cote = T * step
        index = QgsSpatialIndex()
        for i, g in enumerate(self.polygones):
            f = QgsFeature(i)
            f.setGeometry(g)
            index.addFeature(f)
        os.makedirs(os.path.join(self.dossier, "images"), exist_ok=True)
        os.makedirs(os.path.join(self.dossier, "masks"), exist_ok=True)
        nx = math.ceil(self.bb.width() / cote)
        ny = math.ceil(self.bb.height() / cote)
        drv = gdal.GetDriverByName("GTiff")
        wkt = self.source.crs.toWkt()
        for j in range(ny):
            for i in range(nx):
                if self.isCanceled():
                    return False
                x0 = self.bb.xMinimum() + i * cote
                y0 = self.bb.yMaximum() - j * cote
                ext = QgsRectangle(x0, y0 - cote, x0 + cote, y0)
                ids = index.intersects(ext)
                if not ids:
                    continue
                img = self.source.read(ext, T, T)
                masque = np.zeros((T, T), np.uint8)
                for k in ids:
                    g = self.polygones[k]
                    polys = g.asMultiPolygon() if g.isMultipart() else [g.asPolygon()]
                    for poly in polys:
                        for n, anneau in enumerate(poly):
                            pts = np.array([[(p.x() - x0) / step, (y0 - p.y()) / step] for p in anneau], np.int32)
                            cv2.fillPoly(masque, [pts], 1 if n == 0 else 0)
                nom = f"tuile_{j:04d}_{i:04d}.tif"
                gt = (x0, step, 0, y0, 0, -step)
                for sous, arr in (("images", img), ("masks", masque[..., None])):
                    ds = drv.Create(os.path.join(self.dossier, sous, nom), T, T, arr.shape[2], gdal.GDT_Byte,
                                    ["COMPRESS=DEFLATE"])
                    ds.SetGeoTransform(gt)
                    ds.SetProjection(wkt)
                    for b in range(arr.shape[2]):
                        ds.GetRasterBand(b + 1).WriteArray(arr[..., b])
                    ds = None
                self.n += 1
            self.setProgress(100.0 * (j + 1) / ny)
        with open(os.path.join(self.dossier, "LISEZMOI.txt"), "w", encoding="utf-8") as f:
            f.write(f"Jeu d'entraînement BatiScan\nSource : {self.source.name}\nRésolution : {res} m/px\n"
                    f"Taille des vignettes : {T} px\nVignettes : {self.n}\n"
                    "images/ : RVB 8 bits ; masks/ : 0 = fond, 1 = bâtiment\n")
        return True
