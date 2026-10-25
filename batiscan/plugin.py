"""Point d'entrée QGIS : menu, barre d'outils et panneau BatiScan."""

import os

from qgis.PyQt.QtCore import Qt
from qgis.PyQt.QtGui import QIcon
from qgis.PyQt.QtWidgets import QAction, QMessageBox

from . import dependances


class BatiScanPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.dock = None
        self.action = None

    def initGui(self):
        icone = QIcon(os.path.join(os.path.dirname(__file__), "icon.png"))
        self.action = QAction(icone, "BatiScan IA – cartographie des bâtiments", self.iface.mainWindow())
        self.action.setCheckable(True)
        self.action.triggered.connect(self.basculer)
        self.iface.addToolBarIcon(self.action)
        self.iface.addPluginToRasterMenu("BatiScan IA", self.action)

    def unload(self):
        if self.dock:
            self.dock.fermer()
            self.iface.removeDockWidget(self.dock)
            self.dock.deleteLater()
            self.dock = None
        self.iface.removePluginRasterMenu("BatiScan IA", self.action)
        self.iface.removeToolBarIcon(self.action)

    def _verifier_dependances(self):
        manque = dependances.manquantes()
        if not manque:
            return True
        rep = QMessageBox.question(
            self.iface.mainWindow(), "BatiScan IA",
            "Bibliothèques manquantes : " + ", ".join(manque) + ".\n\n"
            "Les installer maintenant avec pip (dans votre profil utilisateur) ?")
        if rep != QMessageBox.StandardButton.Yes:
            return False
        ok, journal = dependances.installer(manque)
        if ok:
            QMessageBox.information(self.iface.mainWindow(), "BatiScan IA",
                                    "Installation terminée. Redémarrez QGIS puis rouvrez BatiScan.")
        else:
            QMessageBox.critical(self.iface.mainWindow(), "BatiScan IA",
                                 "Échec de l'installation :\n\n" + journal[-3000:])
        return False

    def basculer(self, visible):
        if self.dock is None:
            if not self._verifier_dependances():
                self.action.setChecked(False)
                return
            from .ui.dock import DockBatiScan

            self.dock = DockBatiScan(self.iface, self.iface.mainWindow())
            self.iface.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.dock)
            self.dock.visibilityChanged.connect(self.action.setChecked)
        self.dock.setVisible(visible)
