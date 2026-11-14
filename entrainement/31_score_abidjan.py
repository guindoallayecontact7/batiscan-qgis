"""Score par objet de modèles ONNX sur les carrés de test d'Abidjan (orthophoto drone), par la chaîne du plugin.

ORTHO=<orthophoto.tif> python-qgis 31_score_abidjan.py <batiments.gpkg> <carres.gpkg> <non_tranches.gpkg> <sortie.csv> <nom=modele.onnx> ...

Pour chaque carré « test » : détection en mode Bâtiment sur le carré élargi de 20 m ; on garde les objets
dont le centre tombe dans le carré, et les bâtiments de référence dont le centre y tombe.
Les objets détectés à plus de 50 % sur une détection v6 non tranchée par l'utilisateur sont retirés.
Appariement un à un à IoU >= 0,5 (objets pris par score décroissant) : rappel, précision, F1.
Couverture à 30 % : un bâtiment est retrouvé si 30 % de sa surface est couverte ; un objet est juste
si 30 % de sa surface tombe sur la référence.
"""
import csv
import os
import sys
import warnings

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(encoding="utf-8")
from commun_qgis import demarrer_qgis  # noqa: E402

app = demarrer_qgis()
from qgis.core import QgsGeometry, QgsRasterLayer, QgsVectorLayer  # noqa: E402

from batiscan.core.image_source import SourceImage  # noqa: E402
from batiscan.core.tasks import TacheAuto  # noqa: E402

fbat, fcar, fnt, sortie = sys.argv[1:5]
modeles = [a.split("=", 1) for a in sys.argv[5:]]
lay = QgsRasterLayer(os.environ["ORTHO"], "ortho")
src = SourceImage(lay)
ref_all = [f.geometry().makeValid() for f in QgsVectorLayer(fbat, "b", "ogr").getFeatures()]
nt = [f.geometry().makeValid() for f in QgsVectorLayer(fnt, "n", "ogr").getFeatures()]
carres = [(f["nom"], f.geometry()) for f in QgsVectorLayer(fcar, "c", "ogr").getFeatures() if f["role"] == "test"]


def union(gs):
    return QgsGeometry.unaryUnion(gs) if gs else QgsGeometry()


lignes = []
for nom_m, onnx in modeles:
    tot = dict(ref=0, det=0, tp=0, ref_couv=0, det_justes=0)
    for nom_c, cadre in carres:
        ref = [g for g in ref_all if g.centroid().within(cadre)]
        e = cadre.boundingBox()
        e.grow(20)
        t = TacheAuto(src, e, lay.crs(), onnx, {"resolution": None, "seuil": None, "aire_min": 4,
                                                 "regularisation": "batiment", "gpu": True})
        if not t.run():
            raise SystemExit(t.erreur)
        det = []
        for r in t.resultats:
            g = r.geometry().makeValid()
            if not g.centroid().within(cadre):
                continue
            if sum(g.intersection(n).area() for n in nt if n.intersects(g)) > 0.5 * g.area():
                continue
            det.append((r.attributes()[4] if isinstance(r.attributes()[4], float) else 0.0, g))
        det.sort(key=lambda d: -d[0])
        libres = set(range(len(ref)))
        tp = 0
        for _, g in det:
            best, bi = 0.0, None
            for i in libres:
                if ref[i].intersects(g):
                    iou = g.intersection(ref[i]).area() / g.combine(ref[i]).area()
                    if iou > best:
                        best, bi = iou, i
            if best >= 0.5:
                tp += 1
                libres.discard(bi)
        ur, ud = union(ref), union([g for _, g in det])
        rc = sum(1 for g in ref if ud and g.intersection(ud).area() >= 0.3 * g.area())
        dj = sum(1 for _, g in det if ur and g.intersection(ur).area() >= 0.3 * g.area())
        p = tp / len(det) if det else 0
        rr = tp / len(ref) if ref else 0
        lignes.append(dict(modele=nom_m, carre=nom_c, ref=len(ref), det=len(det), tp_iou50=tp,
                           rappel_iou50=round(rr, 3), precision_iou50=round(p, 3),
                           f1_iou50=round(2 * p * rr / (p + rr), 3) if p + rr else 0,
                           rappel_couv30=round(rc / len(ref), 3) if ref else 0,
                           precision_couv30=round(dj / len(det), 3) if det else 0))
        for k, v in (("ref", len(ref)), ("det", len(det)), ("tp", tp), ("ref_couv", rc), ("det_justes", dj)):
            tot[k] += v
        print(lignes[-1], flush=True)
    p, rr = tot["tp"] / max(tot["det"], 1), tot["tp"] / max(tot["ref"], 1)
    lignes.append(dict(modele=nom_m, carre="TOTAL", ref=tot["ref"], det=tot["det"], tp_iou50=tot["tp"],
                       rappel_iou50=round(rr, 3), precision_iou50=round(p, 3),
                       f1_iou50=round(2 * p * rr / (p + rr), 3) if p + rr else 0,
                       rappel_couv30=round(tot["ref_couv"] / max(tot["ref"], 1), 3),
                       precision_couv30=round(tot["det_justes"] / max(tot["det"], 1), 3)))
    print(lignes[-1], flush=True)
with open(sortie, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(lignes[0]), delimiter=";")
    w.writeheader()
    w.writerows(lignes)
