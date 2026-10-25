#!/usr/bin/env bash
# Lance 01_emprises_sam.py pour chaque village et chaque fond, 4 à la fois.
# Usage : bash lancer_emprises.sh <dossier_points> <dossier_sortie>
set -u
PTS="$1"; OUT="$2"; mkdir -p "$OUT"
# chemin court : cmd.exe coupe les guillemets si le .bat et un argument contiennent des espaces
QGIS_PY="${QGIS_PY:-C:/PROGRA~1/QGIS34~1.12/bin/python-qgis-ltr.bat}"
ICI="$(cd "$(dirname "$0")" && pwd)"
export BATISCAN_THREADS=6
export BATISCAN_MODELES="${BATISCAN_MODELES:-$APPDATA/QGIS/QGIS3/profiles/Allaye/batiscan/modeles}"
travail() {
  shp="$1"; fond="$2"
  nom=$(basename "$shp" .shp | sed 's/^Total_building //; s/[^A-Za-z0-9]/_/g')
  sortie="$OUT/${nom}_${fond}.gpkg"
  [ -f "$sortie" ] && { echo "déjà fait : $sortie"; return; }
  ( cd "$ICI" && "$QGIS_PY" 01_emprises_sam.py "$shp" "$fond" "$sortie.tmp.gpkg" 2>&1 | grep -v -i warn ) && mv "$sortie.tmp.gpkg" "$sortie"
}
export -f travail; export OUT ICI QGIS_PY
for fond in google bing; do for shp in "$PTS"/Total_building*.shp; do printf '%s\0%s\0' "$shp" "$fond"; done; done \
  | xargs -0 -n 2 -P 4 bash -c 'travail "$0" "$1"'
echo "TERMINÉ"
