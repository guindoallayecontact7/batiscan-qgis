"""Traitements longs exécutés en tâche de fond (QgsTask).

Chaque tâche reçoit tout ce dont elle a besoin dans le fil principal
(SourceImage déjà construite, points déjà lus) et ne touche à aucune couche du
projet pendant run(). Les couches résultats sont créées dans finished().
"""

import math
import os
import time
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

from . import formes
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
    f.append(QgsField("perimetre_m", QMetaType.Type.Double))
    f.append(QgsField("x_centre", QMetaType.Type.Double))
    f.append(QgsField("y_centre", QMetaType.Type.Double))
    f.append(QgsField("orientation_deg", QMetaType.Type.Double))
    f.append(QgsField("controle", QMetaType.Type.QString))
    return f


# Marquage « a_verifier » des détections automatiques. Seuils mesurés le 13/11/2026 sur 2 013 objets
# jugés (44 zones de Bamako, Kita et villages) avec le modèle v6 : la règle marque 27 % des objets,
# dont 32 % sont faux, contre 6,5 % de faux parmi les autres, et attrape 65 % des fausses détections.
# Le contraste avec le sol (ancienne règle) séparait mal : AUC 0,60 contre 0,74 pour la surface.
# Les seuils de score valent pour v6 ; un autre modèle peut demander d'autres valeurs.
SCORE_A_VERIFIER = 0.85
SURFACE_A_VERIFIER = 20.0  # m²
SURFACE_GRANDE = 400.0  # m² : au-delà, plusieurs toits ou une concession ont pu être fusionnés


def a_verifier(score, surface_m2):
    """Vrai si une détection automatique doit être relue (score faible ou petit objet)."""
    return (score is not None and score < SCORE_A_VERIFIER) or (
        surface_m2 is not None and surface_m2 < SURFACE_A_VERIFIER)


def motifs_controle(f, faible_contraste=False):
    """Raisons de vérifier un bâtiment, séparées par des virgules, ou « ok »."""
    m = []
    if f["score"] is not None and f["score"] < SCORE_A_VERIFIER and f["methode"] == "auto":
        m.append("score_faible")
    s = f["surface_m2"]
    if s is not None and s < SURFACE_A_VERIFIER:
        m.append("petit")
    if f["forme"] == "polygone":
        m.append("forme_irreguliere")
    if faible_contraste:
        m.append("faible_contraste")
    if s is not None and s > SURFACE_GRANDE:
        m.append("grand_fusion_possible")
    return ",".join(m) or "ok"


def completer(f, da, faible_contraste=False):
    """Remplit périmètre, centre, orientation et motifs de contrôle (champs de champs_emprises)."""
    n = f.fields().count()
    a = list(f.attributes())
    f.setAttributes(a + [None] * (n - len(a)))
    g = f.geometry()
    if g is None or g.isEmpty():
        return f
    geo = da.sourceCrs().isGeographic()
    c = g.centroid().asPoint()
    f["perimetre_m"] = round(da.measurePerimeter(g), 1)
    f["x_centre"] = round(c.x(), 7 if geo else 2)
    f["y_centre"] = round(c.y(), 7 if geo else 2)
    # angle du grand côté du rectangle orienté, en degrés depuis le nord, entre 0 et 180
    # (la « hauteur » de QGIS est le côté dans la direction de l'angle)
    _, _, angle, larg, haut = g.orientedMinimumBoundingBox()
    f["orientation_deg"] = round((angle if haut >= larg else angle + 90) % 180, 1) % 180
    f["controle"] = motifs_controle(f, faible_contraste)
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
                completer(f, da)
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
        batiment = p["regularisation"] == "batiment"
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
            def prog(v, nb=nb):
                self.setProgress((nb + v / 100.0) * 100.0 / len(blocs))

            def garde(c, r, bw=bw, bh=bh):
                return marge <= c < marge + bw and marge <= r < marge + bh  # sinon : bloc voisin

            for geom, proba in detecter_bloc(seg, self.source, x0, y0, w, h, step, seuil, amin,
                                             garde, prog, self.isCanceled):
                if batiment:
                    # les formes se construisent à la fin, dans le SCR de l'image
                    feats.append(objet_brut(geom, proba))
                    continue
                geom, forme = regularize(geom, p["regularisation"], tolerance=step)
                feats.append(self._emprise(geom, forme, proba, da, aujourdhui))
        feats = _sans_enceintes(feats)
        if batiment:
            items = [(f.geometry(), f["score"]) for f in feats]
            objets = formes.construire(items, self.upm)
            ecarts = formes.contrastes(self.source, [o[0] for o in objets], self.upm)
            feats = [self._emprise(g, forme, s, da, aujourdhui, e is not None and e < formes.SEUIL_CONTRASTE)
                     for (g, forme, s), e in zip(objets, ecarts)]
        self.resultats = feats
        return True

    def _emprise(self, geom, forme, proba, da, aujourdhui, faible_contraste=False):
        geom = QgsGeometry(geom)
        geom.transform(self.tr)
        f = QgsFeature(champs_emprises())
        f.setGeometry(geom)
        surface = da.measureArea(geom)
        fiabilite = "a_verifier" if a_verifier(proba, surface) else "haute"
        f.setAttributes([None, 0, "auto", fiabilite,
                         round(proba, 3), forme, round(surface, 1),
                         self.source.name, aujourdhui])
        return completer(f, da, faible_contraste)


def objet_brut(geom, proba):
    """Détection brute (SCR de l'image) en attente de la construction des formes."""
    f = QgsFeature(champs_emprises())
    f.setGeometry(geom)
    f.setAttributes([None, 0, "auto", None, proba, "libre", None, None, None, None, None, None, None, None])
    return f


class ImageIncomplete(Exception):
    """Fond en ligne qui renvoie encore des tuiles noires après plusieurs essais."""


PAUSES_RELECTURE_S = (5, 20, 60)
NOIR_TOLERE = 0.005


def lire_complet(source, rect, w, h, annule=None):
    """Lit la fenêtre ; sur un fond en ligne, relit tant qu'une partie revient noire.

    Une tuile Google ou Esri qui n'arrive pas (délai dépassé, limitation du serveur)
    est rendue en noir : sans relecture, la zone passerait pour vide de bâtiments.
    Lève ImageIncomplete si l'image reste trouée après les relectures.
    """
    img = source.read(rect, w, h)
    if not getattr(source, "distant", False):
        return img
    for pause in PAUSES_RELECTURE_S:
        if float((img.max(axis=2) == 0).mean()) <= NOIR_TOLERE:
            return img
        for _ in range(pause):
            if annule and annule():
                return img
            time.sleep(1)
        img = source.read(rect, w, h)
    if float((img.max(axis=2) == 0).mean()) > NOIR_TOLERE:
        raise ImageIncomplete("imagerie en ligne indisponible (tuiles noires)")
    return img


def detecter_bloc(seg, source, x0, y0, w, h, step, seuil, amin_px, garde=None, progress=None, annule=None):
    """Détections brutes d'une fenêtre de w × h pixels de coin haut-gauche (x0, y0).

    Renvoie [(géométrie dans le SCR de l'image, probabilité moyenne)]. `garde(colonne,
    ligne)` reçoit le centre de chaque objet en pixels et dit s'il faut le garder.
    Une fenêtre vide (hors de l'image) ne passe pas par le modèle.
    """
    img = lire_complet(source, QgsRectangle(x0, y0 - h * step, x0 + w * step, y0), w, h, annule)
    if img.max() == 0:
        return []
    prob, interieur = seg.probabilites_image(img, progress, annule)
    lab = etiqueter(prob, interieur, seuil)
    sortie = []
    for _k, rr, cc in instances(lab):
        if len(rr) < amin_px:
            continue
        if garde is not None and not garde(cc.mean(), rr.mean()):
            continue
        haut, gauche = int(rr.min()), int(cc.min())
        hh, ww = int(rr.max()) - haut + 1, int(cc.max()) - gauche + 1
        sub = np.zeros((hh + 2, ww + 2), bool)
        sub[rr - haut + 1, cc - gauche + 1] = True
        geo = (x0 + (gauche - 1) * step, y0 - (haut - 1) * step, step, step)
        proba = float(prob[rr, cc].mean())
        for geom in mask_to_geometries(sub, geo):
            sortie.append((geom, proba))
    return sortie


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
