"""Construit dist/batiscan-<version>.zip, installable dans QGIS ou publiable sur plugins.qgis.org."""

import configparser
import os
import zipfile

RACINE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLUGIN = os.path.join(RACINE, "batiscan")


def main():
    meta = configparser.ConfigParser()
    meta.read(os.path.join(PLUGIN, "metadata.txt"), encoding="utf-8")
    version = meta["general"]["version"]
    os.makedirs(os.path.join(RACINE, "dist"), exist_ok=True)
    sortie = os.path.join(RACINE, "dist", f"batiscan-{version}.zip")
    with zipfile.ZipFile(sortie, "w", zipfile.ZIP_DEFLATED) as z:
        for dossier, sous, fichiers in os.walk(PLUGIN):
            sous[:] = [d for d in sous if d != "__pycache__"]
            for f in fichiers:
                if f.endswith((".pyc", ".pyo")):
                    continue
                chemin = os.path.join(dossier, f)
                z.write(chemin, os.path.join("batiscan", os.path.relpath(chemin, PLUGIN)))
        z.write(os.path.join(RACINE, "LICENSE"), "batiscan/LICENSE")
    print(sortie)


if __name__ == "__main__":
    main()
