"""Construit dist/batiscan-<version>.zip, installable dans QGIS ou publiable sur plugins.qgis.org.

python outils/empaqueter.py                         édition publique (dépôt QGIS)
python outils/empaqueter.py --personnelle <fichier>  édition personnelle : ajoute le catalogue privé
                                                     (dist/batiscan-<version>-personnelle.zip)

L'édition publique ne contient jamais edition_personnelle.json : le script s'arrête s'il en trouve un
dans le dossier de l'extension.
"""

import argparse
import configparser
import os
import zipfile

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN = os.path.join(RACINE, "batiscan")
PRIVE = "edition_personnelle.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--personnelle", help="catalogue privé à ajouter (édition personnelle)")
    a = ap.parse_args()
    if os.path.exists(os.path.join(PLUGIN, PRIVE)):
        raise SystemExit(f"{PRIVE} ne doit pas être dans le dossier de l'extension")
    meta = configparser.ConfigParser()
    meta.read(os.path.join(PLUGIN, "metadata.txt"), encoding="utf-8")
    version = meta["general"]["version"]
    os.makedirs(os.path.join(RACINE, "dist"), exist_ok=True)
    suffixe = "-personnelle" if a.personnelle else ""
    sortie = os.path.join(RACINE, "dist", f"batiscan-{version}{suffixe}.zip")
    with zipfile.ZipFile(sortie, "w", zipfile.ZIP_DEFLATED) as z:
        for dossier, sous, fichiers in os.walk(PLUGIN):
            sous[:] = [d for d in sous if d != "__pycache__"]
            for f in fichiers:
                if f.endswith((".pyc", ".pyo")):
                    continue
                chemin = os.path.join(dossier, f)
                z.write(chemin, os.path.join("batiscan", os.path.relpath(chemin, PLUGIN)))
        z.write(os.path.join(RACINE, "LICENSE"), "batiscan/LICENSE")
        if a.personnelle:
            z.write(a.personnelle, "batiscan/" + PRIVE)
    print(sortie)


if __name__ == "__main__":
    main()
