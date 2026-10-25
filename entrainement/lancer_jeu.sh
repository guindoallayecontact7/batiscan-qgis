#!/usr/bin/env bash
# Construit les vignettes (02_jeu.py) pour chaque emprises/<village>_<fond>.gpkg pas encore traité.
# Usage : bash lancer_jeu.sh <dossier_emprises> <dossier_jeu>
set -u
EMP="$1"; JEU="$2"; mkdir -p "$JEU"
QGIS_PY="${QGIS_PY:-C:/PROGRA~1/QGIS34~1.12/bin/python-qgis-ltr.bat}"
cd "$(dirname "$0")"
for g in "$EMP"/*.gpkg; do
  case "$g" in *.tmp.gpkg) continue;; esac
  nom=$(basename "$g" .gpkg); fond=${nom##*_}
  [ -f "$JEU/$nom/.fait" ] && continue
  "$QGIS_PY" 02_jeu.py "$g" "$fond" "$JEU/$nom" 2>&1 | grep -v -i warn && touch "$JEU/$nom/.fait"
done
