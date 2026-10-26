"""Vérification et installation des bibliothèques Python nécessaires."""

import importlib
import os
import sys

from qgis.PyQt.QtCore import QProcess

REQUIS = {"onnxruntime": "onnxruntime", "cv2": "opencv-python-headless"}


def manquantes():
    out = []
    for module, paquet in REQUIS.items():
        try:
            importlib.import_module(module)
        except Exception:
            out.append(paquet)
    return out


def _python():
    """Interpréteur de QGIS (sys.executable pointe sur qgis.exe sous Windows)."""
    for nom in ("python.exe", "python3.exe", "python3", "python"):
        p = os.path.join(sys.exec_prefix, nom)
        if os.path.isfile(p):
            return p
        p = os.path.join(sys.exec_prefix, "bin", nom)
        if os.path.isfile(p):
            return p
    return sys.executable


def installer(paquets):
    """pip install --user, sans laisser pip changer la version de numpy de QGIS."""
    import tempfile

    import numpy

    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write(f"numpy=={numpy.__version__}\n")
        contraintes = f.name
    # Liste d'arguments fixe (aucune chaîne passée à un shell) ; QProcess
    # n'ouvre pas de fenêtre de console sous Windows.
    proc = QProcess()
    proc.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
    proc.start(_python(), ["-m", "pip", "install", "--user", "-c", contraintes] + list(paquets))
    proc.waitForFinished(-1)
    journal = bytes(proc.readAll()).decode("utf-8", errors="replace")
    os.remove(contraintes)
    ok = proc.exitStatus() == QProcess.ExitStatus.NormalExit and proc.exitCode() == 0
    return ok, journal
