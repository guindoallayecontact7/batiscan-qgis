"""Contrôle qualité : points de terrain contre emprises, et contrôle géométrique des emprises."""

import statistics

from qgis.core import (
    QgsCoordinateTransform,
    QgsDistanceArea,
    QgsFeatureRequest,
    QgsGeometry,
    QgsProject,
    QgsSpatialIndex,
)


def controler(points, polygones, tolerance_m=2.0, selection=False):
    """Compare une couche de points et une couche de polygones.

    Un point est « couvert » s'il tombe dans une emprise ou à moins de
    `tolerance_m` mètres. Renvoie un dict de chiffres et la liste des
    identifiants de points non couverts.
    """
    tr = QgsCoordinateTransform(points.crs(), polygones.crs(), QgsProject.instance().transformContext())
    tol = tolerance_m
    if polygones.crs().isGeographic():
        tol = tolerance_m / 111320.0
    index = QgsSpatialIndex(polygones.getFeatures(QgsFeatureRequest().setNoAttributes()),
                            flags=QgsSpatialIndex.Flag.FlagStoreFeatureGeometries)
    nb_par_poly = {}
    non_couverts = []
    req = QgsFeatureRequest().setNoAttributes()
    feats = points.getSelectedFeatures(req) if selection else points.getFeatures(req)
    n = 0
    for f in feats:
        if not f.hasGeometry():
            continue
        n += 1
        g = QgsGeometry(f.geometry())
        g.transform(tr)
        bb = g.boundingBox().buffered(tol)
        meilleur, dmin = None, None
        for fid in index.intersects(bb):
            d = index.geometry(fid).distance(g)
            if d <= tol and (dmin is None or d < dmin):
                meilleur, dmin = fid, d
        if meilleur is None:
            non_couverts.append(f.id())
        else:
            nb_par_poly[meilleur] = nb_par_poly.get(meilleur, 0) + 1
    nb_poly = polygones.featureCount()
    couverts = n - len(non_couverts)
    return {
        "points": n,
        "points_couverts": couverts,
        "taux_couverture": couverts / n if n else 0.0,
        "emprises": nb_poly,
        "emprises_avec_point": len(nb_par_poly),
        "emprises_sans_point": nb_poly - len(nb_par_poly),
        "emprises_multi_points": sum(1 for v in nb_par_poly.values() if v > 1),
    }, non_couverts


def rapport_texte(r, nom_points, nom_poly):
    return (
        f"Points : {nom_points} — Emprises : {nom_poly}\n"
        f"Points couverts par une emprise : {r['points_couverts']} / {r['points']} "
        f"({100 * r['taux_couverture']:.1f} %)\n"
        f"Emprises contenant au moins un point : {r['emprises_avec_point']} / {r['emprises']}\n"
        f"Emprises sans point (bâtiment non relevé ou fausse détection) : {r['emprises_sans_point']}\n"
        f"Emprises avec plusieurs points (bâtiments fusionnés ?) : {r['emprises_multi_points']}"
    )


SURFACE_MIN_M2 = 4.0  # plus petit, c'est un débris de vectorisation
CHEVAUCHEMENT_MIN_M2 = 0.01
IOU_DOUBLON = 0.9


def controle_geometrique(couche, selection=False):
    """Contrôle d'une couche d'emprises : validité, chevauchements, doublons, débris, motifs.

    Renvoie un dict de chiffres et un dict {fid: [anomalies]}. Les objets ne sont jamais modifiés.
    """
    da = QgsDistanceArea()
    da.setSourceCrs(couche.crs(), QgsProject.instance().transformContext())
    da.setEllipsoid("EPSG:7030")
    feats = list(couche.getSelectedFeatures() if selection else couche.getFeatures())
    geoms = {f.id(): f.geometry() for f in feats if f.hasGeometry() and not f.geometry().isEmpty()}
    anomalies = {}

    def noter(fid, motif):
        anomalies.setdefault(fid, []).append(motif)

    surfaces = {}
    for fid, g in geoms.items():
        if not g.isGeosValid():
            noter(fid, "geometrie_invalide")
        if g.isMultipart() and len(g.asGeometryCollection()) > 1:
            noter(fid, "multi_parties")
        surfaces[fid] = da.measureArea(g)
        if surfaces[fid] < SURFACE_MIN_M2:
            noter(fid, "debris")
    index = QgsSpatialIndex()
    for fid, g in geoms.items():
        index.addFeature(fid, g.boundingBox())
    paires = doublons = 0
    surf_chev = 0.0
    for fid, g in geoms.items():
        if not g.isGeosValid():
            continue
        for autre in index.intersects(g.boundingBox()):
            if autre <= fid or not geoms[autre].isGeosValid() or not g.intersects(geoms[autre]):
                continue
            inter = g.intersection(geoms[autre])
            a = da.measureArea(inter) if inter and not inter.isEmpty() else 0.0
            if a < CHEVAUCHEMENT_MIN_M2:
                continue
            union = surfaces[fid] + surfaces[autre] - a
            if union > 0 and a / union >= IOU_DOUBLON:
                doublons += 1
                noter(autre, "doublon")
            else:
                paires += 1
                surf_chev += a
                noter(fid, "chevauchement")
                noter(autre, "chevauchement")
    motifs = {}
    idx_ctrl = couche.fields().indexOf("controle")
    idx_fiab = couche.fields().indexOf("fiabilite")
    fiab = {}
    for f in feats:
        if idx_ctrl >= 0 and f[idx_ctrl]:
            for m in str(f[idx_ctrl]).split(","):
                motifs[m] = motifs.get(m, 0) + 1
        if idx_fiab >= 0:
            fiab[f[idx_fiab]] = fiab.get(f[idx_fiab], 0) + 1
    s = list(surfaces.values())
    return {
        "objets": len(feats),
        "sans_geometrie": len(feats) - len(geoms),
        "invalides": sum(1 for v in anomalies.values() if "geometrie_invalide" in v),
        "multi_parties": sum(1 for v in anomalies.values() if "multi_parties" in v),
        "debris": sum(1 for v in anomalies.values() if "debris" in v),
        "paires_chevauchantes": paires,
        "surface_chevauchee_m2": surf_chev,
        "doublons": doublons,
        "surface_totale_m2": sum(s),
        "surface_mediane_m2": statistics.median(s) if s else 0.0,
        "motifs": motifs,
        "fiabilite": fiab,
    }, anomalies


def rapport_geometrique(r, nom):
    n = max(r["objets"], 1)
    lignes = [
        f"Couche : {nom} — {r['objets']} objets",
        f"Surface bâtie : {r['surface_totale_m2']:.0f} m² (médiane {r['surface_mediane_m2']:.1f} m² par objet)",
        "",
        "Topologie",
        f"  Géométries invalides : {r['invalides']} ({100 * r['invalides'] / n:.2f} %)",
        f"  Sans géométrie : {r['sans_geometrie']}",
        f"  Objets en plusieurs parties : {r['multi_parties']}",
        f"  Paires qui se chevauchent : {r['paires_chevauchantes']} ({r['surface_chevauchee_m2']:.1f} m²)",
        f"  Doublons (IoU ≥ {IOU_DOUBLON}) : {r['doublons']}",
        f"  Débris (< {SURFACE_MIN_M2:.0f} m²) : {r['debris']}",
    ]
    if r["fiabilite"]:
        lignes += ["", "Fiabilité"] + [f"  {k or '(vide)'} : {v} ({100 * v / n:.1f} %)"
                                      for k, v in sorted(r["fiabilite"].items(), key=lambda x: -x[1])]
    if r["motifs"]:
        lignes += ["", "Motifs de contrôle (un objet peut en avoir plusieurs)"] + [
            f"  {k} : {v} ({100 * v / n:.1f} %)" for k, v in sorted(r["motifs"].items(), key=lambda x: -x[1])]
    return "\n".join(lignes)
