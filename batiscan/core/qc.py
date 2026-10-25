"""Contrôle qualité : points de terrain contre emprises (détectées ou dessinées)."""

from qgis.core import (
    QgsCoordinateTransform,
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
