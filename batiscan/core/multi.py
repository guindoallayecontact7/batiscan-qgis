"""Comptage des bâtiments sur plusieurs localités en une seule tâche.

Trois façons de délimiter une localité :
- un polygone (limite fournie) : on compte les bâtiments dont le centre est dedans ;
- un point et un rayon, recentré sur le village : si le point ne tombe pas dans
  un village, on le déplace vers le sud, le nord, l'est puis l'ouest, par pas
  réguliers, jusqu'à rencontrer du bâti, puis on se place au centre de ce village ;
- un point et un rayon de recherche : chaque village qui touche ce rayon est
  compté à part, avec sa limite (prise dans une couche de limites, ou tracée
  d'après le bâti détecté).

La détection se fait à la demande, par cellules d'une grille fixe dans le SCR de
l'image. Une cellule n'est traitée qu'une fois : deux localités voisines partagent
les mêmes bâtiments et une zone déjà vue ne repasse pas par le modèle.
"""

import math
import traceback
from datetime import date

from qgis.core import (
    QgsCoordinateTransform,
    QgsFeature,
    QgsField,
    QgsFields,
    QgsGeometry,
    QgsPoint,
    QgsPointXY,
    QgsProject,
    QgsRectangle,
    QgsSpatialIndex,
)
from qgis.PyQt.QtCore import QMetaType

from . import formes
from .segmenter import OnnxSegmenter
from .tasks import (ImageIncomplete, _mesureur, _sans_enceintes, _TacheBase, champs_emprises, detecter_bloc,
                    objet_brut)
from .vectorize import regularize

CELLULE_PX = 2048
STATUT_IMAGE = "imagerie en ligne indisponible, à relancer"
DIRECTIONS = (("le sud", 0, -1), ("le nord", 0, 1), ("l'est", 1, 0), ("l'ouest", -1, 0))
MARGE_LIMITE_M = 10.0  # autour du bâti, pour tracer la limite d'un village détecté
ETENDUE_VILLAGE_M = 3000.0  # au-delà, on arrête d'agrandir un village (ville continue)

MODE_RECENTRER = "recentrer"
MODE_LIMITES = "limites"


def champs_zones():
    f = QgsFields()
    for nom, t in (("localite", QMetaType.Type.QString), ("id_localite", QMetaType.Type.QString),
                   ("limite", QMetaType.Type.QString), ("type_zone", QMetaType.Type.QString),
                   ("nb_batiments", QMetaType.Type.Int), ("nb_a_verifier", QMetaType.Type.Int),
                   ("surface_ha", QMetaType.Type.Double), ("densite_ha", QMetaType.Type.Double),
                   ("rayon_m", QMetaType.Type.Double), ("decalage", QMetaType.Type.QString),
                   ("decalage_m", QMetaType.Type.Double), ("statut", QMetaType.Type.QString)):
        f.append(QgsField(nom, t))
    return f


def champs_batiments():
    f = champs_emprises()
    f.append(QgsField("localite", QMetaType.Type.QString))
    return f


def champs_centres():
    f = QgsFields()
    for nom, t in (("localite", QMetaType.Type.QString), ("role", QMetaType.Type.QString),
                   ("decalage", QMetaType.Type.QString)):
        f.append(QgsField(nom, t))
    return f


class Detections:
    """Bâtiments détectés à la demande, cellule par cellule (SCR de l'image)."""

    def __init__(self, source, seg, params, upm_ref):
        self.source = source
        self.seg = seg
        self.res = params["resolution"] or seg.resolution_m
        self.seuil = params["seuil"] if params["seuil"] is not None else seg.threshold
        self.aire_min = params["aire_min"]
        self.reg = params["regularisation"]
        self.cote = CELLULE_PX * self.res * upm_ref
        self.faites = set()
        self.objets = []  # (géométrie, forme, score, a_verifier)
        self.centres = []
        self.index = QgsSpatialIndex()
        self.villages = {}  # id d'objet -> frozenset du village
        self.annule = None
        self.cellules_traitees = 0

    # -- grille -----------------------------------------------------------
    def couvrir(self, rect):
        c = self.cote
        for i in range(math.floor(rect.xMinimum() / c), math.floor(rect.xMaximum() / c) + 1):
            for j in range(math.floor(rect.yMinimum() / c), math.floor(rect.yMaximum() / c) + 1):
                if (i, j) in self.faites:
                    continue
                if self.annule and self.annule():
                    return
                self._cellule(i, j)
                self.faites.add((i, j))

    def _cellule(self, i, j):
        c = self.cote
        coeur = QgsRectangle(i * c, j * c, (i + 1) * c, (j + 1) * c)
        if not coeur.intersects(self.source.extent):
            return
        upm = self.source.units_per_meter(coeur.center())
        step = self.res * upm
        n = int(math.ceil(c / step))
        m = self.seg.tile // 2
        x0, y0 = i * c - m * step, (j + 1) * c + m * step
        brut = detecter_bloc(self.seg, self.source, x0, y0, n + 2 * m, n + 2 * m, step, self.seuil,
                             self.aire_min / (self.res * self.res), None, None, self.annule)
        self.cellules_traitees += 1
        if not brut:
            return
        if self.reg == "batiment":
            # formes construites sur toute la fenêtre, marge comprise, puis on ne garde
            # que les objets dont le centre tombe dans la cellule : les voisines feront le reste
            feats = _sans_enceintes([objet_brut(g, s) for g, s in brut])
            objets = formes.construire([(f.geometry(), f["score"]) for f in feats], upm)
            ecarts = formes.contrastes(self.source, [o[0] for o in objets], upm)
            objets = [(g, forme, s, e is not None and e < formes.SEUIL_CONTRASTE)
                      for (g, forme, s), e in zip(objets, ecarts)]
        else:
            feats = _sans_enceintes([objet_brut(g, s) for g, s in brut])
            objets = []
            for f in feats:
                g, forme = regularize(f.geometry(), self.reg, tolerance=step)
                objets.append((g, forme, f["score"], False))
        for g, forme, s, av in objets:
            p = g.pointOnSurface().asPoint()
            if not (coeur.xMinimum() <= p.x() < coeur.xMaximum() and coeur.yMinimum() <= p.y() < coeur.yMaximum()):
                continue
            k = len(self.objets)
            self.objets.append((g, forme, s, av))
            self.centres.append(p)
            ft = QgsFeature(k)
            ft.setGeometry(QgsGeometry.fromPointXY(p))
            self.index.addFeature(ft)

    # -- requêtes ---------------------------------------------------------
    def proches(self, p, d):
        """Objets dont le centre est à moins de d (unités du SCR) de p, du plus proche au plus loin."""
        r = QgsRectangle(p.x() - d, p.y() - d, p.x() + d, p.y() + d)
        self.couvrir(r)
        ids = [(math.hypot(self.centres[k].x() - p.x(), self.centres[k].y() - p.y()), k)
               for k in self.index.intersects(r)]
        return [k for dist, k in sorted(ids) if dist <= d]

    def dans(self, zone):
        """Objets dont le centre est dans la géométrie `zone`."""
        bb = zone.boundingBox()
        self.couvrir(bb)
        moteur = QgsGeometry.createGeometryEngine(zone.constGet())
        moteur.prepareGeometry()
        return [k for k in self.index.intersects(bb)
                if moteur.contains(QgsPoint(self.centres[k].x(), self.centres[k].y()))]

    def village(self, graine, ecart, etendue):
        """Objets reliés à `graine` de proche en proche (écart et étendue en unités du SCR)."""
        if graine in self.villages:
            return self.villages[graine]
        o = self.centres[graine]
        membres = {graine}
        pile = [graine]
        while pile:
            if self.annule and self.annule():
                break
            p = self.centres[pile.pop()]
            for k in self.proches(p, ecart):
                if k not in membres and math.hypot(self.centres[k].x() - o.x(), self.centres[k].y() - o.y()) <= etendue:
                    membres.add(k)
                    pile.append(k)
        v = frozenset(membres)
        for k in v:
            self.villages.setdefault(k, v)
        return v

    def centre(self, ids):
        n = len(ids)
        return QgsPointXY(sum(self.centres[k].x() for k in ids) / n, sum(self.centres[k].y() for k in ids) / n)

    def limite(self, ids, marge):
        """Limite d'un groupe de bâtiments : enveloppe concave (ou convexe) élargie de `marge`."""
        geoms = QgsGeometry.collectGeometry([self.objets[k][0] for k in ids])
        env = None
        try:
            env = geoms.concaveHull(0.3)
        except Exception:
            env = None
        if env is None or env.isEmpty() or env.area() <= 0:
            env = geoms.convexHull()
        return env.buffer(marge, 8)


def _texte_decalage(de, dn):
    morceaux = []
    if abs(dn) >= 1:
        morceaux.append(f"{'Nord' if dn > 0 else 'Sud'} {abs(dn):.0f} m")
    if abs(de) >= 1:
        morceaux.append(f"{'Est' if de > 0 else 'Ouest'} {abs(de):.0f} m")
    return ", ".join(morceaux) or "aucun"


class TacheMulti(_TacheBase):
    """Comptage des bâtiments localité par localité, en un seul lancement."""

    def __init__(self, source, localites, limites, crs_sortie, chemin_onnx, params):
        """localites : [{"nom", "id", "geom" (SCR de l'image), "point" (bool)}] ;
        limites : [(nom, géométrie dans le SCR de l'image)] ou liste vide."""
        super().__init__("BatiScan : comptage multi-localités")
        self.source = source
        self.localites = localites
        self.limites = limites
        self.crs_sortie = crs_sortie
        self.tr = QgsCoordinateTransform(source.crs, crs_sortie, QgsProject.instance().transformContext())
        self.chemin = chemin_onnx
        self.p = params
        xs = [loc["geom"].boundingBox().center().x() for loc in localites]
        ys = [loc["geom"].boundingBox().center().y() for loc in localites]
        self.upm_ref = source.units_per_meter(QgsPointXY(sum(xs) / len(xs), sum(ys) / len(ys)))
        self.zones = []  # QgsFeature (champs_zones), SCR de sortie
        self.batiments = []  # QgsFeature (champs_batiments), SCR de sortie
        self.centres = []  # QgsFeature (champs_centres), SCR de sortie
        self.synthese = {}
        self.resolution = None

    def run(self):
        try:
            return self._run()
        except Exception:
            self.erreur = traceback.format_exc()
            return False

    # ------------------------------------------------------------------
    def _run(self):
        p = self.p
        seg = OnnxSegmenter(self.chemin, p["gpu"])
        det = Detections(self.source, seg, p, self.upm_ref)
        det.annule = self.isCanceled
        self.det = det
        self.resolution = det.res
        self.da = _mesureur(self.crs_sortie)
        self.idx_limites = QgsSpatialIndex()
        for i, (_nom, g) in enumerate(self.limites):
            ft = QgsFeature(i)
            ft.setGeometry(g)
            self.idx_limites.addFeature(ft)
        compte = {}  # id d'objet -> localité (première qui le compte)
        n = len(self.localites)
        for i, loc in enumerate(self.localites):
            if self.isCanceled():
                return False
            try:
                if not loc["point"]:
                    zones = [self._zone_fournie(loc, loc["geom"], "", "limite fournie")]
                elif p["mode"] == MODE_RECENTRER:
                    zones = self._recentrer(loc)
                else:
                    zones = self._chercher_limites(loc)
            except ImageIncomplete:
                # pas de faux zéro : la localité est signalée, ses cellules seront relues au prochain passage
                g = loc["geom"] if not loc["point"] else loc["geom"].buffer(
                    p["rayon_m"] * self.source.units_per_meter(loc["geom"].asPoint()), 32)
                zones = [{"geom": g, "ids": [], "limite": "", "type": "non traitée", "statut": STATUT_IMAGE,
                          "rayon": None, "decalage": "", "decalage_m": None}]
            if self.isCanceled():
                return False
            for z in zones:
                self._enregistrer(loc, z, compte)
            self.setProgress(100.0 * (i + 1) / n)
        aujourdhui = date.today().isoformat()
        for k, nom in compte.items():
            g, forme, s, av = det.objets[k]
            geom = QgsGeometry(g)
            geom.transform(self.tr)
            f = QgsFeature(champs_batiments())
            f.setGeometry(geom)
            fiab = "a_verifier" if av else "haute" if s >= 0.6 else "moyenne"
            f.setAttributes([None, 0, "auto", fiab, round(s, 3), forme, round(self.da.measureArea(geom), 1),
                             self.source.name, aujourdhui, nom])
            self.batiments.append(f)
        non_traitees = sum(1 for z in self.zones if z["statut"] == STATUT_IMAGE)
        sans = sum(1 for z in self.zones if z["nb_batiments"] == 0) - non_traitees
        self.synthese = {
            "localites": n,
            "zones": len(self.zones),
            "batiments": sum(z["nb_batiments"] for z in self.zones),
            "uniques": len(compte),
            "a_verifier": sum(z["nb_a_verifier"] for z in self.zones),
            "zones_vides": sans,
            "non_traitees": non_traitees,
            "cellules": det.cellules_traitees,
        }
        return True

    # -- zones ------------------------------------------------------------
    def _zone_fournie(self, loc, geom, limite, type_zone, statut="", rayon=None, decalage="", decalage_m=None):
        return {"geom": geom, "ids": self.det.dans(geom), "limite": limite, "type": type_zone,
                "statut": statut, "rayon": rayon, "decalage": decalage, "decalage_m": decalage_m}

    def _limites_touchant(self, zone):
        ids = []
        for i in self.idx_limites.intersects(zone.boundingBox()):
            if self.limites[i][1].intersects(zone):
                ids.append(i)
        return ids

    def _village_en(self, q, upm):
        p = self.p
        for k in self.det.proches(q, p["ecart_m"] * upm):
            v = self.det.village(k, p["ecart_m"] * upm, ETENDUE_VILLAGE_M * upm)
            if len(v) >= p["min_batiments"]:
                return v
        return None

    def _recentrer(self, loc):
        p = self.p
        pt = loc["geom"].asPoint()
        upm = self.source.units_per_meter(pt)
        # Le point tombe dans une limite connue : on la prend telle quelle.
        if self.limites:
            pg = QgsGeometry.fromPointXY(pt)
            for i in self._limites_touchant(pg):
                nom, g = self.limites[i]
                return [self._zone_fournie(loc, g, nom, "limite (couche)", "point dans la limite")]
        v = self._village_en(pt, upm)
        trouve = "point dans le village" if v else None
        if v is None:
            d = p["pas_m"]
            while v is None and d <= p["decalage_max_m"] + 1e-6:
                for nom_dir, dx, dy in DIRECTIONS:
                    q = QgsPointXY(pt.x() + dx * d * upm, pt.y() + dy * d * upm)
                    v = self._village_en(q, upm)
                    if v:
                        trouve = f"village trouvé vers {nom_dir} à {d:.0f} m"
                        break
                    if self.isCanceled():
                        return []
                d += p["pas_m"]
        self._centre(loc, pt, "point fourni", "")
        if v is None:
            zone = QgsGeometry.fromPointXY(pt).buffer(p["rayon_m"] * upm, 32)
            z = self._zone_fournie(loc, zone, "", "rayon (sans village)",
                                   f"aucun village dans {p['decalage_max_m']:.0f} m autour du point",
                                   p["rayon_m"], "aucun", 0.0)
            return [z]
        c = self.det.centre(v)
        de, dn = (c.x() - pt.x()) / upm, (c.y() - pt.y()) / upm
        rayon = p["rayon_m"]
        if p["couvrir"]:
            cg = QgsGeometry.fromPointXY(c)
            loin = max(self.det.objets[k][0].hausdorffDistance(cg) for k in v) / upm
            rayon = max(rayon, math.ceil(loin + MARGE_LIMITE_M))
        texte = _texte_decalage(de, dn)
        self._centre(loc, c, "centre du village", texte)
        zone = QgsGeometry.fromPointXY(c).buffer(rayon * upm, 64)
        return [self._zone_fournie(loc, zone, "", "rayon recentré", trouve, rayon, texte,
                                   round(math.hypot(de, dn), 1))]

    def _chercher_limites(self, loc):
        p = self.p
        pt = loc["geom"].asPoint()
        upm = self.source.units_per_meter(pt)
        cercle = QgsGeometry.fromPointXY(pt).buffer(p["rayon_recherche_m"] * upm, 64)
        zones = []
        if self.limites:
            for i in self._limites_touchant(cercle):
                nom, g = self.limites[i]
                zones.append(self._zone_fournie(loc, g, nom, "limite (couche)",
                                                f"limite à moins de {p['rayon_recherche_m']:.0f} m"))
            if zones:
                return zones
        # Pas de couche de limites (ou rien dedans) : limites tracées d'après le bâti.
        vus = set()
        n = 0
        for k in self.det.dans(cercle):
            if k in vus:
                continue
            v = self.det.village(k, p["ecart_m"] * upm, ETENDUE_VILLAGE_M * upm)
            vus |= v
            if len(v) < p["min_batiments"]:
                continue
            n += 1
            g = self.det.limite(v, MARGE_LIMITE_M * upm)
            statut = "aucune limite de la couche dans le rayon ; " if self.limites else ""
            zones.append({"geom": g, "ids": sorted(v), "limite": f"village détecté {n}",
                          "type": "limite d'après le bâti", "statut": statut + "tracée d'après les bâtiments détectés",
                          "rayon": None, "decalage": "", "decalage_m": None})
            if self.isCanceled():
                return []
        if not zones:
            zones.append(self._zone_fournie(loc, cercle, "", "rayon (sans village)",
                                            f"aucun village dans {p['rayon_recherche_m']:.0f} m",
                                            p["rayon_recherche_m"]))
        return zones

    # -- sorties ----------------------------------------------------------
    def _centre(self, loc, pt, role, decalage):
        g = QgsGeometry.fromPointXY(pt)
        g.transform(self.tr)
        f = QgsFeature(champs_centres())
        f.setGeometry(g)
        f.setAttributes([loc["nom"], role, decalage])
        self.centres.append(f)

    def _enregistrer(self, loc, z, compte):
        geom = QgsGeometry(z["geom"])
        geom.transform(self.tr)
        ids = z["ids"]
        for k in ids:
            compte.setdefault(k, loc["nom"])
        n = len(ids)
        av = sum(1 for k in ids if self.det.objets[k][3])
        ha = self.da.measureArea(geom) / 10000.0
        f = QgsFeature(champs_zones())
        f.setGeometry(geom)
        f.setAttributes([loc["nom"], str(loc["id"]), z["limite"], z["type"], n, av, round(ha, 2),
                         round(n / ha, 2) if ha > 0 else None, z["rayon"], z["decalage"], z["decalage_m"],
                         z["statut"]])
        self.zones.append(f)
