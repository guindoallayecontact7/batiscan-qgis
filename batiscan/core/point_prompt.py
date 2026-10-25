"""Du point relevé sur le terrain au masque du bâtiment, avec SAM.

Un point de terrain tombe souvent à côté du toit (sur la cour, l'ombre, le
mur d'enceinte). La stratégie :
1. invite directe sur le point, les points voisins servant d'exemples
   négatifs pour ne pas fusionner deux bâtiments ;
2. si le masque est implausible (trop grand, trop petit, en confettis,
   collé au bord), nouvelle invite avec les coins de la vignette en négatif ;
3. sinon, recherche en couronnes autour du point : on garde le masque
   plausible le plus proche du point d'origine.
"""

import math

import cv2
import numpy as np


class Criteres:
    def __init__(self, res_m, aire_min_m2=4.0, aire_max_m2=600.0, solidite_min=0.75,
                 rayon_recherche_m=3.0, dist_max_m=4.0):
        px2 = res_m * res_m
        self.res = res_m
        self.aire_min = aire_min_m2 / px2
        self.aire_max = aire_max_m2 / px2
        self.solidite_min = solidite_min
        self.rayons = [r / res_m for r in (1.2, 2.0, rayon_recherche_m) if r <= rayon_recherche_m] or [rayon_recherche_m / res_m]
        self.dist_max = dist_max_m / res_m


def nettoyer(mask, pc, pr, dist_max):
    """Garde la tache contenant le point (ou la plus proche), trous bouchés.

    Renvoie (masque, distance au point en px, solidité) ou None.
    """
    m = mask.astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if n <= 1:
        return None
    h, w = m.shape
    ic, ir = int(min(max(pc, 0), w - 1)), int(min(max(pr, 0), h - 1))
    k = lab[ir, ic]
    dist = 0.0
    if k == 0:
        ys, xs = np.nonzero(m)
        d = np.hypot(xs - pc, ys - pr)
        j = int(np.argmin(d))
        dist = float(d[j])
        if dist > dist_max:
            return None
        k = lab[ys[j], xs[j]]
    comp = (lab == k).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    plein = np.zeros_like(comp)
    cv2.drawContours(plein, cnts, -1, 1, -1)
    aire = float(plein.sum())
    hull = cv2.convexHull(np.vstack(cnts))
    aire_hull = max(cv2.contourArea(hull), 1.0)
    solidite = min(cv2.contourArea(max(cnts, key=cv2.contourArea)) / aire_hull, 1.0) if aire > 0 else 0
    return plein.astype(bool), dist, solidite


def plausible(clean, crit):
    if clean is None:
        return False
    m, _, sol = clean
    a = m.sum()
    bord = m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()
    return crit.aire_min <= a <= crit.aire_max and sol >= crit.solidite_min and not bord


def segmenter_point(sam, pc, pr, voisins, crit):
    """Masque du bâtiment désigné par le point (pc, pr) de la vignette courante.

    sam doit avoir reçu l'image (set_image). voisins : autres points de terrain
    de la vignette, en pixels. Renvoie un dict (masque, score, methode,
    distance_m) ou None si aucun masque plausible.
    """
    h, w = sam.orig_size
    neg = [tuple(v) for v in voisins]
    coins = [(2, 2), (w - 3, 2), (2, h - 3), (w - 3, h - 3)]

    def essai(qc, qr, extra):
        pts = [(qc, qr)] + neg + extra
        labs = [1] + [0] * (len(neg) + len(extra))
        m, score = sam.predict(pts, labs)[0]
        return nettoyer(m, pc, pr, crit.dist_max), score

    for methode, extra in (("direct", []), ("coins", coins)):
        clean, score = essai(pc, pr, extra)
        if plausible(clean, crit):
            return {"masque": clean[0], "score": score, "methode": methode,
                    "distance_m": clean[1] * crit.res, "solidite": clean[2]}

    meilleur = None
    for rayon in crit.rayons:
        for k in range(8):
            ang = k * math.pi / 4
            qc, qr = pc + rayon * math.cos(ang), pr + rayon * math.sin(ang)
            if not (0 <= qc < w and 0 <= qr < h):
                continue
            clean, score = essai(qc, qr, coins)
            if not plausible(clean, crit):
                continue
            cle = (clean[1], -score)
            if meilleur is None or cle < meilleur[0]:
                meilleur = (cle, clean, score)
        if meilleur:
            break
    if meilleur is None:
        return None
    _, clean, score = meilleur
    return {"masque": clean[0], "score": score, "methode": "recherche",
            "distance_m": clean[1] * crit.res, "solidite": clean[2]}
