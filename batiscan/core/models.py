"""Catalogue des modèles, téléchargement et modèles personnels."""

import hashlib
import json
import os
import zipfile

from qgis.core import QgsApplication, QgsSettings

HF = "https://huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/"

CATALOGUE = [
    {
        "id": "mobilesam",
        "nom": "Pointage rapide · MobileSAM : un clic par toit, tout terrain (11 Mo)",
        "type": "sam",
        "url": HF + "mobile_sam_20230629_quant.zip",
        "taille_mo": 11,
        "licence": "Apache-2.0",
    },
    {
        "id": "sam_vit_b",
        "nom": "Pointage précis · SAM ViT-B : contours plus fidèles, plus lent (75 Mo)",
        "type": "sam",
        "url": HF + "sam_vit_b_01ec64_quant.zip",
        "taille_mo": 75,
        "licence": "Apache-2.0",
    },
    {
        "id": "batiscan_sahel_v6",
        "nom": "Bâti sahélien · BatiScan v6 (recommandé) : villages, bourgs et villes, sépare les maisons accolées (30 cm/px, 98 Mo)",
        "type": "segmenteur",
        "url": "https://github.com/guindoallayecontact7/batiscan-qgis/releases/download/v0.5.0/batiscan_sahel_v6.onnx",
        "fichier": "batiscan_sahel_v6.onnx",
        "sha256": "38f1daf1bfce5d55c6e5d4aebc8b0f4e60f5345677ab3c8625816d8854dd8257",  # pragma: allowlist secret (empreinte publique du modèle)
        "taille_mo": 98,
        "licence": "recherche et tests, non commercial",
    },
    {
        "id": "deepness_ramp",
        "nom": "Bâti général · RAMP (Deepness) : grands bâtiments isolés, fusionne les maisons accolées (50 cm/px, 82 Mo)",
        "type": "segmenteur",
        "url": "https://chmura.put.poznan.pl/s/MwhgQNhyQF3fuBs/download",
        "fichier": "buildings_ramp.onnx",
        "taille_mo": 82,
        "licence": "voir Deepness Model ZOO",
    },
]

_CLE_PERSO = "batiscan/modeles_perso"


def dossier_modeles():
    # BATISCAN_MODELES : utile pour les scripts lancés hors de l'interface QGIS
    d = os.environ.get("BATISCAN_MODELES") or os.path.join(
        QgsApplication.qgisSettingsDirPath(), "batiscan", "modeles")
    os.makedirs(d, exist_ok=True)
    return d


def _yaml_simple(path):
    """Lit un config.yaml plat (clé: valeur), sans dépendance à PyYAML."""
    out = {}
    with open(path, encoding="utf-8") as f:
        for ligne in f:
            if ":" in ligne and not ligne.lstrip().startswith("#"):
                k, v = ligne.split(":", 1)
                out[k.strip()] = v.strip().strip("'\"")
    return out


def chemins(modele):
    """Fichiers locaux d'un modèle, ou None s'il n'est pas installé."""
    if modele.get("perso"):
        return {"onnx": modele["fichier"]} if os.path.exists(modele["fichier"]) else None
    d = os.path.join(dossier_modeles(), modele["id"])
    if modele["type"] == "segmenteur":
        p = os.path.join(d, modele["fichier"])
        return {"onnx": p} if os.path.exists(p) else None
    cfg = os.path.join(d, "config.yaml")
    if not os.path.exists(cfg):
        return None
    c = _yaml_simple(cfg)
    enc = os.path.join(d, c.get("encoder_model_path", ""))
    dec = os.path.join(d, c.get("decoder_model_path", ""))
    if not (os.path.isfile(enc) and os.path.isfile(dec)):
        return None
    canevas = (int(c.get("max_height", 682)), int(c.get("max_width", 1024)))
    return {"encodeur": enc, "decodeur": dec, "canevas": canevas}


def _empreinte(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for bloc in iter(lambda: f.read(1 << 20), b""):
            h.update(bloc)
    return h.hexdigest()


def installer_archive(modele, fichier_telecharge):
    """Range le fichier téléchargé dans le dossier du modèle (dézippe si besoin)."""
    attendu = modele.get("sha256")
    if attendu and _empreinte(fichier_telecharge) != attendu:
        os.remove(fichier_telecharge)
        raise ValueError("le fichier reçu ne correspond pas au modèle publié, il a été supprimé")
    d = os.path.join(dossier_modeles(), modele["id"])
    os.makedirs(d, exist_ok=True)
    if zipfile.is_zipfile(fichier_telecharge):
        with zipfile.ZipFile(fichier_telecharge) as z:
            for nom in z.namelist():
                cible = os.path.normpath(os.path.join(d, os.path.basename(nom)))
                if not nom.endswith("/") and cible.startswith(os.path.normpath(d)):
                    with z.open(nom) as src, open(cible, "wb") as dst:
                        dst.write(src.read())
        os.remove(fichier_telecharge)
    else:
        os.replace(fichier_telecharge, os.path.join(d, modele["fichier"]))


def modeles_perso():
    brut = QgsSettings().value(_CLE_PERSO, "[]")
    try:
        chemins_ = json.loads(brut)
    except ValueError:
        chemins_ = []
    return [{"id": "perso:" + p, "nom": _nom_perso(p), "type": "segmenteur",
             "fichier": p, "perso": True} for p in chemins_]


def _nom_perso(p):
    """Nom d'un modèle ajouté par l'utilisateur ; signale s'il double un modèle du catalogue."""
    nom = "Personnel · " + os.path.splitext(os.path.basename(p))[0]
    try:
        taille = os.path.getsize(p)
    except OSError:
        return nom + " (fichier introuvable)"
    for m in CATALOGUE:
        if m["type"] == "segmenteur":
            q = os.path.join(dossier_modeles(), m["id"], m["fichier"])
            if os.path.isfile(q) and os.path.getsize(q) == taille and _meme_fichier(p, q):
                return f"{nom} : copie de « {m['nom'].split(' :')[0]} »"
    return nom + " : ajouté à la main, non mesuré"


def _meme_fichier(a, b, n=1 << 20):
    """Compare le premier et le dernier Mo (les versions d'un même réseau ont la même taille)."""
    with open(a, "rb") as fa, open(b, "rb") as fb:
        if fa.read(n) != fb.read(n):
            return False
        fa.seek(-n, 2)
        fb.seek(-n, 2)
        return fa.read(n) == fb.read(n)


def ajouter_modele_perso(path):
    actuels = [m["fichier"] for m in modeles_perso()]
    if path not in actuels:
        actuels.append(path)
        QgsSettings().setValue(_CLE_PERSO, json.dumps(actuels))


def retirer_modele_perso(path):
    """Retire un modèle personnel de la liste ; le fichier reste sur le disque."""
    restants = [m["fichier"] for m in modeles_perso() if m["fichier"] != path]
    QgsSettings().setValue(_CLE_PERSO, json.dumps(restants))


def tous(type_=None):
    return [m for m in CATALOGUE + modeles_perso() if type_ is None or m["type"] == type_]
