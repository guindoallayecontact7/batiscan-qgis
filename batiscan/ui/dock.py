"""Panneau principal de BatiScan."""

import os
import tempfile

from qgis.core import (
    Qgis,
    QgsApplication,
    QgsCategorizedSymbolRenderer,
    QgsCoordinateTransform,
    QgsFeatureRequest,
    QgsFileDownloader,
    QgsFillSymbol,
    QgsGeometry,
    QgsProject,
    QgsRendererCategory,
    QgsSettings,
    QgsWkbTypes,
)
from qgis.gui import QgsExtentWidget, QgsFileWidget, QgsMapLayerComboBox
from qgis.PyQt.QtCore import Qt, QUrl
from qgis.PyQt.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..core import models, qc
from ..core.image_source import SourceImage
from ..core.runtime import available_providers
from ..core.sam_onnx import SamOnnx
from ..core.tasks import TacheAuto, TacheJeuEntrainement, TachePoints, champs_emprises, couche_memoire
from .outil_clic import OutilClic

REGULARISATIONS = [("auto", "Auto (rectangle, cercle ou contour)"), ("rectangle", "Rectangle orienté"),
                   ("simplifie", "Contour simplifié"), ("brut", "Contour brut")]

AIDE_POINTS = (
    "Pour chaque point relevé, SAM détoure le toit voisin. Les points proches "
    "servent d'exemples négatifs pour ne pas fusionner deux bâtiments. Le champ "
    "« fiabilite » indique la méthode : haute = point sur le toit, faible = toit "
    "trouvé en cherchant autour du point."
)


def _spin(val, mini, maxi, pas=1.0, dec=2, suffixe=""):
    s = QDoubleSpinBox()
    s.setRange(mini, maxi)
    s.setDecimals(dec)
    s.setSingleStep(pas)
    s.setValue(val)
    if suffixe:
        s.setSuffix(" " + suffixe)
    return s


def _combo_couche(filtre):
    c = QgsMapLayerComboBox()
    c.setFilters(filtre)
    return c


def _style_emprises(couche):
    couleurs = {"haute": "#1a9850", "moyenne": "#fd8d3c", "faible": "#d73027"}
    cats = []
    for val, coul in couleurs.items():
        sym = QgsFillSymbol.createSimple({"color": "0,0,0,0", "outline_color": coul, "outline_width": "0.5"})
        cats.append(QgsRendererCategory(val, sym, val))
    couche.setRenderer(QgsCategorizedSymbolRenderer("fiabilite", cats))


class DockBatiScan(QDockWidget):
    def __init__(self, iface, parent=None):
        super().__init__("BatiScan IA – bâtiments", parent)
        self.setObjectName("BatiScanDock")
        self.iface = iface
        self._taches = []
        self._telechargements = {}
        self._outil = None
        self._couche_saisie = None
        self._sam_interactif = None

        tabs = QTabWidget()
        tabs.addTab(self._onglet_points(), "Points → emprises")
        tabs.addTab(self._onglet_clic(), "Clic")
        tabs.addTab(self._onglet_auto(), "Automatique")
        tabs.addTab(self._onglet_qc(), "Contrôle")
        tabs.addTab(self._onglet_jeu(), "Entraînement")
        tabs.addTab(self._onglet_modeles(), "Modèles")
        self.setWidget(tabs)

    # ------------------------------------------------------------------ utils
    def _scroll(self, w):
        s = QScrollArea()
        s.setWidgetResizable(True)
        s.setWidget(w)
        s.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        for c in w.findChildren(QComboBox):
            c.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            c.setMinimumWidth(120)
        return s

    def _combo_modeles(self, type_):
        c = QComboBox()
        for m in models.tous(type_):
            c.addItem(m["nom"], m)
        return c

    def _rafraichir_combos(self):
        for c, t in ((self.pt_modele, "sam"), (self.clic_modele, "sam"), (self.auto_modele, "segmenteur")):
            actuel = c.currentText()
            c.clear()
            for m in models.tous(t):
                c.addItem(m["nom"], m)
            i = c.findText(actuel)
            if i >= 0:
                c.setCurrentIndex(i)
        self._remplir_liste_modeles()

    def _gpu(self):
        return QgsSettings().value("batiscan/gpu", True, type=bool)

    def _msg(self, texte, niveau=Qgis.MessageLevel.Info, duree=6):
        self.iface.messageBar().pushMessage("BatiScan", texte, niveau, duree)

    def _modele_pret(self, modele):
        """Chemins du modèle ; propose le téléchargement s'il manque."""
        ch = models.chemins(modele)
        if ch:
            return ch
        if modele.get("perso"):
            self._msg("Fichier introuvable : " + modele["fichier"], Qgis.MessageLevel.Warning)
            return None
        rep = QMessageBox.question(
            self, "BatiScan",
            f"Le modèle « {modele['nom']} » n'est pas encore installé "
            f"(environ {modele['taille_mo']} Mo, licence {modele['licence']}).\n"
            f"Source : {modele['url']}\n\nLe télécharger maintenant ?")
        if rep == QMessageBox.StandardButton.Yes:
            self._telecharger(modele)
        return None

    def _lancer(self, tache, suite):
        tache.termine.connect(suite)
        self._taches.append(tache)
        QgsApplication.taskManager().addTask(tache)
        self._msg(tache.description() + " : lancé (suivi dans la barre des tâches en bas).")

    def _fin_tache(self, tache):
        if tache in self._taches:
            self._taches.remove(tache)
        if tache.erreur:
            QMessageBox.critical(self, "BatiScan", "Le traitement a échoué :\n\n" + tache.erreur[-2000:])
            return False
        if tache.isCanceled():
            self._msg("Traitement annulé.", Qgis.MessageLevel.Warning)
            return False
        return True

    # ----------------------------------------------------- points → emprises
    def _onglet_points(self):
        w = QWidget()
        f = QFormLayout(w)
        aide = QLabel(AIDE_POINTS)
        aide.setWordWrap(True)
        f.addRow(aide)
        self.pt_raster = _combo_couche(Qgis.LayerFilter.RasterLayer)
        self.pt_points = _combo_couche(Qgis.LayerFilter.PointLayer)
        self.pt_selection = QCheckBox("Sélection uniquement")
        self.pt_modele = self._combo_modeles("sam")
        self.pt_res = _spin(0.3, 0.05, 2.0, 0.05, 2, "m/px")
        self.pt_vignette = QSpinBox()
        self.pt_vignette.setRange(96, 512)
        self.pt_vignette.setSingleStep(32)
        self.pt_vignette.setValue(192)
        self.pt_vignette.setSuffix(" px")
        self.pt_vignette.setToolTip("Contexte vu par SAM autour de chaque point. "
                                    "Trop grand : les petites cases sont ignorées.")
        self.pt_voisins = _spin(15, 0, 50, 1, 0, "m")
        self.pt_recherche = _spin(3, 0, 10, 0.5, 1, "m")
        self.pt_amin = _spin(4, 0, 100, 1, 0, "m²")
        self.pt_amax = _spin(600, 20, 10000, 50, 0, "m²")
        self.pt_reg = QComboBox()
        for k, t in REGULARISATIONS:
            self.pt_reg.addItem(t, k)
        btn = QPushButton("Lancer")
        btn.clicked.connect(self._lancer_points)
        f.addRow("Image", self.pt_raster)
        f.addRow("Points bâtiments", self.pt_points)
        f.addRow("", self.pt_selection)
        f.addRow("Modèle SAM", self.pt_modele)
        f.addRow("Résolution de travail", self.pt_res)
        f.addRow("Taille de vignette", self.pt_vignette)
        f.addRow("Rayon des voisins (négatifs)", self.pt_voisins)
        f.addRow("Recherche autour du point", self.pt_recherche)
        f.addRow("Surface minimale", self.pt_amin)
        f.addRow("Surface maximale", self.pt_amax)
        f.addRow("Forme", self.pt_reg)
        f.addRow(btn)
        return self._scroll(w)

    def _lancer_points(self):
        raster, pts = self.pt_raster.currentLayer(), self.pt_points.currentLayer()
        if not raster or not pts:
            self._msg("Choisissez une image et une couche de points.", Qgis.MessageLevel.Warning)
            return
        ch = self._modele_pret(self.pt_modele.currentData())
        if not ch:
            return
        source = SourceImage(raster)
        tr = QgsCoordinateTransform(pts.crs(), source.crs, QgsProject.instance())
        req = QgsFeatureRequest().setNoAttributes()
        it = pts.getSelectedFeatures(req) if self.pt_selection.isChecked() else pts.getFeatures(req)
        points = []
        for ft in it:
            if ft.hasGeometry():
                g = QgsGeometry(ft.geometry())
                g.transform(tr)
                points.append((ft.id(), g.centroid().asPoint()))
        if not points:
            self._msg("Aucun point à traiter.", Qgis.MessageLevel.Warning)
            return
        params = {"resolution": self.pt_res.value(), "vignette": self.pt_vignette.value(),
                  "rayon_voisins": self.pt_voisins.value(), "rayon_recherche": self.pt_recherche.value(),
                  "aire_min": self.pt_amin.value(), "aire_max": self.pt_amax.value(),
                  "regularisation": self.pt_reg.currentData(), "gpu": self._gpu()}
        tache = TachePoints(source, points, pts.crs(), ch, params)
        tache._couche_points = pts
        self._lancer(tache, self._fin_points)

    def _fin_points(self, tache):
        if not self._fin_tache(tache):
            return
        pts = tache._couche_points
        couche = couche_memoire(f"Emprises – {pts.name()}", tache.crs_sortie, "Polygon",
                                champs_emprises(), tache.resultats)
        _style_emprises(couche)
        QgsProject.instance().addMapLayer(couche)
        if tache.echecs:
            req = QgsFeatureRequest().setFilterFids(tache.echecs)
            nt = couche_memoire(f"Points sans emprise – {pts.name()}", pts.crs(),
                                QgsWkbTypes.displayString(pts.wkbType()), pts.fields(), pts.getFeatures(req))
            QgsProject.instance().addMapLayer(nt)
        s = tache.stats
        n = len(tache.points)
        self._msg(f"{len(tache.resultats)} emprises pour {n} points — directes {s['direct']}, "
                  f"coins {s['coins']}, recherche {s['recherche']}, doublons {s['doublon']}, "
                  f"sans emprise {len(tache.echecs)}.", Qgis.MessageLevel.Success, 15)

    # ------------------------------------------------------------------ clic
    def _onglet_clic(self):
        w = QWidget()
        f = QFormLayout(w)
        aide = QLabel("Cliquez au centre d'un toit pour ajouter son emprise dans la couche "
                      "« BatiScan – saisie ». Clic droit : annule le dernier ajout. "
                      "Pratique pour compléter les points sans emprise.")
        aide.setWordWrap(True)
        f.addRow(aide)
        self.clic_raster = _combo_couche(Qgis.LayerFilter.RasterLayer)
        self.clic_modele = self._combo_modeles("sam")
        self.clic_res = _spin(0.3, 0.05, 2.0, 0.05, 2, "m/px")
        self.clic_reg = QComboBox()
        for k, t in REGULARISATIONS:
            self.clic_reg.addItem(t, k)
        self.clic_btn = QPushButton("Activer l'outil de clic")
        self.clic_btn.setCheckable(True)
        self.clic_btn.toggled.connect(self._basculer_outil)
        f.addRow("Image", self.clic_raster)
        f.addRow("Modèle SAM", self.clic_modele)
        f.addRow("Résolution", self.clic_res)
        f.addRow("Forme", self.clic_reg)
        f.addRow(self.clic_btn)
        return self._scroll(w)

    def _contexte_clic(self):
        raster = self.clic_raster.currentLayer()
        if not raster:
            self._msg("Choisissez une image.", Qgis.MessageLevel.Warning)
            return None
        modele = self.clic_modele.currentData()
        ch = models.chemins(modele)
        if not ch:
            self.clic_btn.setChecked(False)
            self._modele_pret(modele)
            return None
        cle = (raster.id(), modele["id"], self._gpu())
        if self._sam_interactif is None or self._sam_interactif[0] != cle:
            self._sam_interactif = (cle, SourceImage(raster),
                                    SamOnnx(ch["encodeur"], ch["decodeur"], ch.get("canevas", (682, 1024)), self._gpu()))
        if self._couche_saisie is None or QgsProject.instance().mapLayer(self._couche_saisie.id()) is None:
            crs = QgsProject.instance().crs()
            if crs.isGeographic():
                crs = self._sam_interactif[1].crs
            self._couche_saisie = couche_memoire("BatiScan – saisie", crs, "Polygon", champs_emprises())
            _style_emprises(self._couche_saisie)
            QgsProject.instance().addMapLayer(self._couche_saisie)
        return {"source": self._sam_interactif[1], "sam": self._sam_interactif[2], "couche": self._couche_saisie,
                "params": {"vignette": 192, "resolution": self.clic_res.value(), "aire_min": 2.0,
                           "aire_max": 2000.0, "regularisation": self.clic_reg.currentData()}}

    def _basculer_outil(self, actif):
        canvas = self.iface.mapCanvas()
        if actif:
            if self._outil is None:
                self._outil = OutilClic(self.iface, self._contexte_clic)
            canvas.setMapTool(self._outil)
            self.clic_btn.setText("Désactiver l'outil de clic")
        else:
            if self._outil and canvas.mapTool() is self._outil:
                canvas.unsetMapTool(self._outil)
            self.clic_btn.setText("Activer l'outil de clic")

    # ------------------------------------------------------------ automatique
    def _onglet_auto(self):
        w = QWidget()
        f = QFormLayout(w)
        aide = QLabel("Détection sans points sur une emprise, avec un modèle de segmentation ONNX "
                      "(format Deepness). Le modèle RAMP fourni est générique : pour le bâti rural "
                      "sahélien, entraînez votre propre modèle (onglet Entraînement).")
        aide.setWordWrap(True)
        f.addRow(aide)
        self.auto_raster = _combo_couche(Qgis.LayerFilter.RasterLayer)
        self.auto_modele = self._combo_modeles("segmenteur")
        self.auto_emprise = QgsExtentWidget()
        self.auto_emprise.setMapCanvas(self.iface.mapCanvas())
        self.auto_emprise.setOriginalExtent(self.iface.mapCanvas().extent(),
                                            self.iface.mapCanvas().mapSettings().destinationCrs())
        self.auto_emprise.setCurrentExtent(self.iface.mapCanvas().extent(),
                                           self.iface.mapCanvas().mapSettings().destinationCrs())
        self.auto_res = _spin(0, 0, 5, 0.05, 2, "m/px")
        self.auto_res.setSpecialValueText("celle du modèle")
        self.auto_seuil = _spin(0, 0, 1, 0.05, 2)
        self.auto_seuil.setSpecialValueText("celui du modèle")
        self.auto_amin = _spin(6, 0, 200, 1, 0, "m²")
        self.auto_reg = QComboBox()
        for k, t in REGULARISATIONS:
            self.auto_reg.addItem(t, k)
        btn = QPushButton("Lancer la détection")
        btn.clicked.connect(self._lancer_auto)
        f.addRow("Image", self.auto_raster)
        f.addRow("Modèle", self.auto_modele)
        f.addRow("Emprise", self.auto_emprise)
        f.addRow("Résolution", self.auto_res)
        f.addRow("Seuil de probabilité", self.auto_seuil)
        f.addRow("Surface minimale", self.auto_amin)
        f.addRow("Forme", self.auto_reg)
        f.addRow(btn)
        return self._scroll(w)

    def _lancer_auto(self):
        raster = self.auto_raster.currentLayer()
        if not raster:
            self._msg("Choisissez une image.", Qgis.MessageLevel.Warning)
            return
        ch = self._modele_pret(self.auto_modele.currentData())
        if not ch:
            return
        source = SourceImage(raster)
        ext = self.auto_emprise.outputExtent()
        tr = QgsCoordinateTransform(self.auto_emprise.outputCrs(), source.crs, QgsProject.instance())
        ext = tr.transformBoundingBox(ext)
        crs_sortie = QgsProject.instance().crs()
        if crs_sortie.isGeographic() or not crs_sortie.isValid():
            crs_sortie = source.crs
        params = {"resolution": self.auto_res.value() or None, "seuil": self.auto_seuil.value() or None,
                  "aire_min": self.auto_amin.value(), "regularisation": self.auto_reg.currentData(),
                  "gpu": self._gpu()}
        largeur_km = ext.width() / source.units_per_meter(ext.center()) / 1000
        hauteur_km = ext.height() / source.units_per_meter(ext.center()) / 1000
        if largeur_km * hauteur_km > 25:
            rep = QMessageBox.question(self, "BatiScan", f"L'emprise couvre {largeur_km * hauteur_km:.0f} km². "
                                       "Le traitement peut être long. Continuer ?")
            if rep != QMessageBox.StandardButton.Yes:
                return
        tache = TacheAuto(source, ext, crs_sortie, ch["onnx"], params)
        self._lancer(tache, self._fin_auto)

    def _fin_auto(self, tache):
        if not self._fin_tache(tache):
            return
        couche = couche_memoire(f"Détection – {tache.source.name}", tache.crs_sortie, "Polygon",
                                champs_emprises(), tache.resultats)
        _style_emprises(couche)
        QgsProject.instance().addMapLayer(couche)
        self._msg(f"{len(tache.resultats)} bâtiments détectés (résolution {tache.resolution} m/px).",
                  Qgis.MessageLevel.Success, 10)

    # --------------------------------------------------------------- contrôle
    def _onglet_qc(self):
        w = QWidget()
        f = QFormLayout(w)
        self.qc_points = _combo_couche(Qgis.LayerFilter.PointLayer)
        self.qc_poly = _combo_couche(Qgis.LayerFilter.PolygonLayer)
        self.qc_tol = _spin(2, 0, 20, 0.5, 1, "m")
        self.qc_couche = QCheckBox("Couche des points non couverts")
        self.qc_couche.setChecked(True)
        self.qc_txt = QPlainTextEdit()
        self.qc_txt.setReadOnly(True)
        btn = QPushButton("Comparer")
        btn.clicked.connect(self._lancer_qc)
        f.addRow("Points de terrain", self.qc_points)
        f.addRow("Emprises", self.qc_poly)
        f.addRow("Tolérance", self.qc_tol)
        f.addRow("", self.qc_couche)
        f.addRow(btn)
        f.addRow(self.qc_txt)
        return self._scroll(w)

    def _lancer_qc(self):
        pts, poly = self.qc_points.currentLayer(), self.qc_poly.currentLayer()
        if not pts or not poly:
            return
        r, manquants = qc.controler(pts, poly, self.qc_tol.value())
        self.qc_txt.setPlainText(qc.rapport_texte(r, pts.name(), poly.name()))
        if self.qc_couche.isChecked() and manquants:
            req = QgsFeatureRequest().setFilterFids(manquants)
            nt = couche_memoire(f"Non couverts – {pts.name()}", pts.crs(),
                                QgsWkbTypes.displayString(pts.wkbType()), pts.fields(), pts.getFeatures(req))
            QgsProject.instance().addMapLayer(nt)

    # ----------------------------------------------------------- entraînement
    def _onglet_jeu(self):
        w = QWidget()
        f = QFormLayout(w)
        aide = QLabel("Exporte des vignettes image + masque à partir d'emprises vérifiées. "
                      "Ce jeu sert à entraîner un modèle adapté à votre zone (voir le dossier "
                      "« entrainement » du dépôt), à recharger ensuite dans l'onglet Modèles.")
        aide.setWordWrap(True)
        f.addRow(aide)
        self.jeu_raster = _combo_couche(Qgis.LayerFilter.RasterLayer)
        self.jeu_poly = _combo_couche(Qgis.LayerFilter.PolygonLayer)
        self.jeu_filtre = QCheckBox("Fiabilité « haute » uniquement")
        self.jeu_filtre.setChecked(True)
        self.jeu_res = _spin(0.3, 0.05, 2, 0.05, 2, "m/px")
        self.jeu_taille = QSpinBox()
        self.jeu_taille.setRange(128, 1024)
        self.jeu_taille.setSingleStep(64)
        self.jeu_taille.setValue(256)
        self.jeu_taille.setSuffix(" px")
        self.jeu_dossier = QgsFileWidget()
        self.jeu_dossier.setStorageMode(QgsFileWidget.StorageMode.GetDirectory)
        btn = QPushButton("Exporter")
        btn.clicked.connect(self._lancer_jeu)
        f.addRow("Image", self.jeu_raster)
        f.addRow("Emprises", self.jeu_poly)
        f.addRow("", self.jeu_filtre)
        f.addRow("Résolution", self.jeu_res)
        f.addRow("Taille des vignettes", self.jeu_taille)
        f.addRow("Dossier de sortie", self.jeu_dossier)
        f.addRow(btn)
        return self._scroll(w)

    def _lancer_jeu(self):
        raster, poly = self.jeu_raster.currentLayer(), self.jeu_poly.currentLayer()
        dossier = self.jeu_dossier.filePath()
        if not raster or not poly or not dossier:
            self._msg("Choisissez l'image, les emprises et le dossier.", Qgis.MessageLevel.Warning)
            return
        source = SourceImage(raster)
        tr = QgsCoordinateTransform(poly.crs(), source.crs, QgsProject.instance())
        req = QgsFeatureRequest()
        if self.jeu_filtre.isChecked() and poly.fields().indexOf("fiabilite") >= 0:
            req.setFilterExpression("\"fiabilite\" = 'haute'")
        geoms = []
        for ft in poly.getFeatures(req):
            if ft.hasGeometry():
                g = QgsGeometry(ft.geometry())
                g.transform(tr)
                geoms.append(g)
        if not geoms:
            self._msg("Aucune emprise à exporter.", Qgis.MessageLevel.Warning)
            return
        tache = TacheJeuEntrainement(source, geoms, dossier,
                                     {"resolution": self.jeu_res.value(), "taille": self.jeu_taille.value()})
        self._lancer(tache, self._fin_jeu)

    def _fin_jeu(self, tache):
        if self._fin_tache(tache):
            self._msg(f"{tache.n} vignettes écrites dans {tache.dossier}.", Qgis.MessageLevel.Success, 10)

    # ---------------------------------------------------------------- modèles
    def _onglet_modeles(self):
        w = QWidget()
        v = QVBoxLayout(w)
        self.liste_modeles = QListWidget()
        v.addWidget(QLabel("Modèles (stockés dans le profil QGIS) :"))
        v.addWidget(self.liste_modeles)
        h = QVBoxLayout()
        b1 = QPushButton("Télécharger")
        b1.clicked.connect(self._telecharger_selection)
        b2 = QPushButton("Ajouter un modèle .onnx…")
        b2.clicked.connect(self._ajouter_perso)
        b3 = QPushButton("Ouvrir le dossier")
        b3.clicked.connect(lambda: os.startfile(models.dossier_modeles()) if hasattr(os, "startfile") else None)
        for b in (b1, b2, b3):
            h.addWidget(b)
        v.addLayout(h)
        self.cb_gpu = QCheckBox("Utiliser le GPU si possible")
        self.cb_gpu.setChecked(self._gpu())
        self.cb_gpu.toggled.connect(lambda x: QgsSettings().setValue("batiscan/gpu", x))
        v.addWidget(self.cb_gpu)
        prov = QLabel("Moteurs onnxruntime : " + ", ".join(available_providers()))
        prov.setWordWrap(True)
        v.addWidget(prov)
        self._remplir_liste_modeles()
        return w

    def _remplir_liste_modeles(self):
        self.liste_modeles.clear()
        for m in models.tous():
            etat = "installé" if models.chemins(m) else "à télécharger"
            it = QListWidgetItem(f"{m['nom']} — {m['type']} — {etat}")
            it.setData(256, m)
            self.liste_modeles.addItem(it)

    def _telecharger_selection(self):
        it = self.liste_modeles.currentItem()
        if it:
            m = it.data(256)
            if not m.get("perso") and not models.chemins(m):
                self._telecharger(m)

    def _telecharger(self, modele):
        if modele["id"] in self._telechargements:
            return
        fd, tmp = tempfile.mkstemp(suffix=".part", dir=models.dossier_modeles())
        os.close(fd)
        dl = QgsFileDownloader(QUrl(modele["url"]), tmp, "", True)

        def fini():
            try:
                models.installer_archive(modele, tmp)
                self._msg(f"Modèle installé : {modele['nom']}", Qgis.MessageLevel.Success)
            except Exception as exc:
                self._msg(f"Installation impossible : {exc}", Qgis.MessageLevel.Critical)
            self._telechargements.pop(modele["id"], None)
            self._rafraichir_combos()

        def erreur(erreurs):
            self._telechargements.pop(modele["id"], None)
            self._msg("Téléchargement échoué : " + "; ".join(erreurs), Qgis.MessageLevel.Critical, 10)

        def progression(recu, total):
            if total > 0:
                self.iface.statusBarIface().showMessage(
                    f"BatiScan : {modele['nom']} {100 * recu / total:.0f} %", 2000)

        dl.downloadCompleted.connect(fini)
        dl.downloadError.connect(erreur)
        dl.downloadProgress.connect(progression)
        self._telechargements[modele["id"]] = dl
        dl.startDownload()
        self._msg(f"Téléchargement de {modele['nom']} en cours…")

    def _ajouter_perso(self):
        path, _ = QFileDialog.getOpenFileName(self, "Modèle de segmentation ONNX", "", "ONNX (*.onnx)")
        if path:
            models.ajouter_modele_perso(path)
            self._rafraichir_combos()

    def fermer(self):
        if self._outil:
            self.iface.mapCanvas().unsetMapTool(self._outil)
