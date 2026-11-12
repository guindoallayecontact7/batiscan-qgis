"""Import d'une liste de localités et export du tableau de comptage."""

import re

from qgis.core import (
    QgsCoordinateReferenceSystem,
    QgsFeature,
    QgsGeometry,
    QgsPointXY,
    QgsProject,
    QgsVectorFileWriter,
    QgsVectorLayer,
)

_LON = re.compile(r"^(lon|lng|long|x|coord_?x|easting)$|longitude", re.I)
_LAT = re.compile(r"^(lat|y|coord_?y|northing)$|latitude", re.I)
_NOM = ("village", "localit", "nom", "name", "site")


def ouvrir(chemin, nom):
    """Couche OGR du fichier. Un CSV qui n'est pas en UTF-8 (export Excel Windows) est lu en Windows-1252."""
    lyr = QgsVectorLayer(chemin, nom, "ogr")
    if lyr.isValid() and chemin.lower().endswith((".csv", ".txt")):
        with open(chemin, "rb") as f:
            octets = f.read()
        try:
            octets.decode("utf-8")
        except UnicodeDecodeError:
            lyr.setProviderEncoding("windows-1252")
    return lyr


def champ_nom(couche):
    """Nom du champ qui ressemble le plus à un nom de localité, ou ''."""
    if couche is None:
        return ""
    for fld in couche.fields():
        n = fld.name().lower()
        if any(m in n for m in _NOM):
            return fld.name()
    return ""


def _nombre(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).strip().replace(",", "."))
    except ValueError:
        return None


def colonnes_coordonnees(champs):
    """(colonne longitude/X, colonne latitude/Y) d'un tableau, None si absente."""
    noms = [f.name() for f in champs]
    lon = next((n for n in noms if _LON.search(n)), None)
    lat = next((n for n in noms if _LAT.search(n)), None)
    return lon, lat


def points_depuis_tableau(table, nom, crs_projet):
    """Couche de points en mémoire depuis un tableau (Excel, CSV) avec colonnes de coordonnées.

    Renvoie (couche, None) ou (None, message d'erreur). Coordonnées en degrés : EPSG:4326 ;
    sinon le SCR du projet. Les lignes sans coordonnées lisibles sont ignorées.
    """
    lon, lat = colonnes_coordonnees(table.fields())
    if not lon or not lat:
        noms = ", ".join(f.name() for f in table.fields())
        return None, ("Colonnes de coordonnées introuvables. Nommez-les par exemple « longitude » et "
                      f"« latitude » (ou « x » et « y »).\n\nColonnes du fichier : {noms}")
    lignes = []
    for ft in table.getFeatures():
        x, y = _nombre(ft[lon]), _nombre(ft[lat])
        if x is not None and y is not None:
            lignes.append((x, y, ft.attributes()))
    if not lignes:
        return None, f"Aucune ligne avec des coordonnées lisibles dans « {lon} » et « {lat} »."
    degres = all(-180 <= x <= 180 and -90 <= y <= 90 for x, y, _ in lignes)
    crs = QgsCoordinateReferenceSystem("EPSG:4326") if degres else crs_projet
    couche = QgsVectorLayer(f"Point?crs={crs.authid() or 'EPSG:4326'}", nom, "memory")
    pr = couche.dataProvider()
    pr.addAttributes(table.fields().toList())
    couche.updateFields()
    feats = []
    for x, y, attrs in lignes:
        f = QgsFeature(couche.fields())
        f.setGeometry(QgsGeometry.fromPointXY(QgsPointXY(x, y)))
        f.setAttributes(attrs)
        feats.append(f)
    pr.addFeatures(feats)
    couche.updateExtents()
    return couche, None


def exporter_tableau(zones, chemin, csv):
    """Écrit le tableau des zones (sans géométrie) suivi d'une ligne TOTAL. Renvoie un message d'erreur ou None."""
    table = QgsVectorLayer("None", "comptage", "memory")
    table.dataProvider().addAttributes(zones.fields().toList())
    table.updateFields()
    feats = []
    total = {"nb_batiments": 0, "nb_a_verifier": 0, "surface_ha": 0.0}
    for ft in zones.getFeatures():
        f = QgsFeature(table.fields())
        f.setAttributes(ft.attributes())
        feats.append(f)
        for k in total:
            total[k] += ft[k] or 0
    t = QgsFeature(table.fields())
    t["localite"] = "TOTAL"
    t["nb_batiments"] = total["nb_batiments"]
    t["nb_a_verifier"] = total["nb_a_verifier"]
    t["surface_ha"] = round(total["surface_ha"], 2)
    if total["surface_ha"] > 0:
        t["densite_ha"] = round(total["nb_batiments"] / total["surface_ha"], 2)
    t["statut"] = f"{len(feats)} zones"
    feats.append(t)
    table.dataProvider().addFeatures(feats)
    opts = QgsVectorFileWriter.SaveVectorOptions()
    opts.driverName = "CSV" if csv else "XLSX"
    opts.fileEncoding = "UTF-8"
    if csv:
        opts.layerOptions = ["SEPARATOR=SEMICOLON"]
    err = QgsVectorFileWriter.writeAsVectorFormatV3(table, chemin, QgsProject.instance().transformContext(), opts)
    if err[0] != QgsVectorFileWriter.WriterError.NoError:
        return str(err[1]) or "erreur d'écriture"
    return None
