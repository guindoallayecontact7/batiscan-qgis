"""Catalogue des modèles, téléchargement et modèles personnels."""

import json
import os
import zipfile

from qgis.core import QgsApplication, QgsSettings

HF = "https://huggingface.co/vietanhdev/segment-anything-onnx-models/resolve/main/"

CATALOGUE = [
    {
        "id": "mobilesam",
        "nom": "MobileSAM quantifié (11 Mo, rapide)",
        "type": "sam",
        "url": HF + "mobile_sam_20230629_quant.zip",
        "taille_mo": 11,
        "licence": "Apache-2.0",
    },
    {
        "id": "sam_vit_b",
        "nom": "SAM ViT-B quantifié (75 Mo, plus précis)",
        "type": "sam",
        "url": HF + "sam_vit_b_01ec64_quant.zip",
        "taille_mo": 75,
        "licence": "Apache-2.0",
    },
    {
        "id": "batiscan_sahel_v2",
        "nom": "BatiScan Sahel v2 – bâti rural et bourgs (98 Mo, 30 cm/px)",
        "type": "segmenteur",
        "url": "https://github.com/guindoallayecontact7/batiscan-qgis/releases/download/v0.3.0/batiscan_sahel_v2.onnx",
        "fichier": "batiscan_sahel_v2.onnx",
        "taille_mo": 98,
        "licence": "recherche et tests, non commercial",
    },
    {
        "id": "batiscan_sahel_v1",
        "nom": "BatiScan Sahel v1 – ancienne version (98 Mo, 30 cm/px)",
        "type": "segmenteur",
        "url": "https://github.com/guindoallayecontact7/batiscan-qgis/releases/download/v0.2.0/batiscan_sahel_v1.onnx",
        "fichier": "batiscan_sahel_v1.onnx",
        "taille_mo": 98,
        "licence": "recherche et tests, non commercial",
    },
    {
        "id": "deepness_ramp",
        "nom": "Bâtiments RAMP – Deepness (82 Mo, 50 cm/px)",
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


def installer_archive(modele, fichier_telecharge):
    """Range le fichier téléchargé dans le dossier du modèle (dézippe si besoin)."""
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
    return [{"id": "perso:" + p, "nom": os.path.basename(p) + " (personnel)", "type": "segmenteur",
             "fichier": p, "perso": True} for p in chemins_]


def ajouter_modele_perso(path):
    actuels = [m["fichier"] for m in modeles_perso()]
    if path not in actuels:
        actuels.append(path)
        QgsSettings().setValue(_CLE_PERSO, json.dumps(actuels))


def tous(type_=None):
    return [m for m in CATALOGUE + modeles_perso() if type_ is None or m["type"] == type_]
